"""Train-only hard-task injection with reservations across asynchronous batches."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from random import Random
from threading import Lock
from uuid import uuid4

PROTOCOL = "shopping-train-only-feedback-v1"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_feedback_config(config):
    options = config.get("shopping_feedback_sampling", {})
    if not options.get("enable", False):
        return
    flash = config.get("shopping_flashreinforce", {})
    if not flash.get("enable") or flash.get("execution") != "async" or flash.get("invalid_batch_policy") != "skip_batch":
        raise ValueError("feedback sampling requires async FlashREINFORCE whole-batch admission")
    fraction = options.get("hard_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not 0 <= fraction < 1:
        raise ValueError("hard_fraction must be finite and in [0, 1)")
    if not options.get("manifest"):
        raise ValueError("feedback sampling requires a frozen manifest")
    if config["trainer"].get("resume_mode") != "disable":
        raise ValueError("feedback sampling requires a fresh controller; sampler resume is unsupported")


def load_manifest(path, train_files, *, root=None):
    """Bind ID-only sampling metadata to unchanged parquet and frozen split IDs."""
    import pyarrow.parquet as pq

    root = Path(root or os.environ.get("SHOPPING_GRPO_ROOT") or Path(__file__).resolve().parents[4])
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(manifest) != {"protocol", "train_sha256", "train_task_ids", "hard_task_ids"} or manifest["protocol"] != PROTOCOL:
        raise ValueError("unexpected feedback manifest schema (IDs only)")
    paths = [train_files] if isinstance(train_files, (str, Path)) else list(train_files)
    if len(paths) != 1 or sha256(paths[0]) != manifest["train_sha256"]:
        raise ValueError("feedback training parquet fingerprint differs")
    actual = [int(r["extra_info"]["task_id"]) for r in pq.read_table(paths[0], columns=["extra_info"]).to_pylist()]
    train, hard = manifest["train_task_ids"], manifest["hard_task_ids"]
    if any(type(t) is not int for t in train + hard):
        raise ValueError("feedback task IDs must be integers")
    if actual != train or len(set(train)) != len(train) or len(set(hard)) != len(hard) or not hard or not set(hard) <= set(train):
        raise ValueError("feedback task identity mismatch or duplicate IDs")
    def ids(name):
        return {int(json.loads(line)["task_id"]) for line in (root / name).read_text(encoding="utf-8").splitlines() if line.strip()}
    if not set(train) <= ids("data/grpo/train.jsonl") or set(train) & (ids("data/grpo/validation.jsonl") | ids("data/evaluation/tasks.jsonl")):
        raise ValueError("feedback manifest violates frozen train/validation/Final isolation")
    return manifest


class ReservationSampler:
    """Uniform background plus a quota-controlled hard lane, with fresh rollouts.

    A reservation lasts through worker completion, FIFO residence and learner
    consumption. Thus completion order cannot create duplicate tasks in a batch.
    The background includes hard tasks: injection fraction is NOT total hard share.
    """

    def __init__(self, task_ids, hard_ids, *, hard_fraction, seed, limit):
        self.tasks, self.hard = tuple(task_ids), frozenset(hard_ids)
        if len(set(self.tasks)) != len(self.tasks) or not self.hard or not self.hard <= set(self.tasks):
            raise ValueError("sampler requires unique train IDs and a nonempty hard subset")
        if not 0 <= hard_fraction < 1 or limit <= 0:
            raise ValueError("invalid sampler bounds")
        self.fraction = Fraction(str(hard_fraction)).limit_denominator(10000)
        self.rng, self.limit = Random(seed), limit
        self.reserved, self.draws = set(), 0
        self.lock = Lock()

    def draw(self):
        with self.lock:
            if self.draws >= self.limit:
                raise StopIteration
            lane = "hard" if int((self.draws + 1) * self.fraction) > int(self.draws * self.fraction) else "background"
            available = [t for t in self.tasks if t not in self.reserved and (lane == "background" or t in self.hard)]
            if not available:
                return None  # Producer must drain completions, not block inside next().
            task = self.rng.choice(available)
            self.reserved.add(task)
            self.draws += 1
            return {"task_id": task, "lane": lane, "hard_member": task in self.hard, "dispatch_index": self.draws}

    def release(self, task_ids):
        with self.lock:
            if len(set(task_ids)) != len(task_ids) or not set(task_ids) <= self.reserved:
                raise ValueError("releasing an absent or duplicate task reservation")
            self.reserved.difference_update(task_ids)


class FeedbackSource:
    """Use native filtered dataset/collation; never add annotations to model inputs."""

    def __init__(self, trainer):
        import numpy as np
        from verl import DataProto

        validate_feedback_config(trainer.config)
        options = trainer.config.shopping_feedback_sampling
        manifest = load_manifest(options.manifest, trainer.config.data.train_files)
        dataset = trainer.train_dataset
        actual = [int(row["task_id"]) for row in dataset.dataframe["extra_info"]]
        if len(actual) != len(set(actual)) or set(actual) != set(manifest["train_task_ids"]):
            raise ValueError("native filtered dataset differs from frozen feedback train pool")
        self.trainer, self.dataset = trainer, dataset
        self.indices = {task: i for i, task in enumerate(actual)}
        self.sampler = ReservationSampler(manifest["train_task_ids"], manifest["hard_task_ids"],
            hard_fraction=options.hard_fraction, seed=int(trainer.config.data.seed),
            limit=int(trainer.config.shopping_flashreinforce.max_candidate_batches) * int(trainer.config.data.train_batch_size))
        # More distinct tasks than all reservations guarantees the requested lane
        # remains available, even with a full completed FIFO and held learner batch.
        bound = int(trainer.config.shopping_flashreinforce.async_inflight) + int(trainer.config.data.train_batch_size) * (int(trainer.config.shopping_flashreinforce.async_queue_batches) + 1)
        if len(manifest["hard_task_ids"]) <= bound or len(actual) <= bound:
            raise ValueError("feedback pools are too small for async reservation bounds")
        self.path = Path(trainer.config.trainer.default_local_dir) / "feedback_sampling.jsonl"
        if self.path.exists():
            raise ValueError("feedback audit already exists; use a fresh output directory")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock, self.closed = Lock(), False
        self.counts = {}
        self.cancel_requested = set()
        self.np, self.proto = np, DataProto
        self.record("configuration", manifest_sha256=sha256(options.manifest), hard_fraction=options.hard_fraction,
                    hard_pool=len(manifest["hard_task_ids"]), train_pool=len(actual),
                    reservation_bound=bound, native_sampler_resume_supported=False)

    def __iter__(self):
        return self

    def try_next(self):
        choice = self.sampler.draw()
        if choice is None:
            return None
        task = choice["task_id"]
        batch = self.proto.from_single_dict(self.trainer.train_dataloader.collate_fn([self.dataset[self.indices[task]]]))
        batch.meta_info["temperature"] = 1.0
        uid = str(uuid4())
        batch.non_tensor_batch["uid"] = self.np.array([uid], dtype=object)
        choice = dict(choice, uid=uid)
        generation = self.trainer._get_gen_batch(batch)
        self.record("dispatch", choice)
        return batch, generation, choice

    def record(self, stage, choice=None, **payload):
        with self.lock:
            if self.closed:
                return
            if choice is not None:
                if stage == "cancel_requested":
                    self.cancel_requested.add(choice["uid"])
                counts = self.counts.setdefault(stage, Counter())
                counts["total"] += 1
                counts["hard_lane"] += choice["lane"] == "hard"
                counts["hard_member"] += choice["hard_member"]
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"protocol": PROTOCOL, "stage": stage, "choice": choice, **payload}, sort_keys=True) + "\n")

    def on_cancel(self, item):
        self.record("cancel_requested", item[2])

    def worker_done(self, choice, future):
        failed = future.cancelled() or future.exception() is not None
        with self.lock:
            intentional = choice["uid"] in self.cancel_requested
        self.record("cancelled" if failed and intentional else "finished", choice,
                    failed=failed and not intentional)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            with self.sampler.lock:
                state = {"counts": self.counts, "draws": self.sampler.draws, "reserved_task_ids": sorted(self.sampler.reserved)}
            self.path.with_suffix(".state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
