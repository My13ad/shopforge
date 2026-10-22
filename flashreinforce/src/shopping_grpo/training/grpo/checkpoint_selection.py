"""Validation-only checkpoint choice and stopping at committed update boundaries."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from shopping_grpo.training.grpo.diagnostics import capture_validation, strict_success

PROTOCOL = "validation50-serial-eager-native-greedy-v1"


@dataclass
class SelectionPolicy:
    patience: int = 3
    minimum_updates: int = 128
    best_step: int | None = None
    best_successes: int = -1
    last_step: int = -1
    non_improving_evaluations: int = 0

    def __post_init__(self):
        if (isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 1
                or isinstance(self.minimum_updates, bool) or not isinstance(self.minimum_updates, int)
                or self.minimum_updates < 0):
            raise ValueError("selection requires positive patience and nonnegative minimum_updates")

    def observe(self, step, successes):
        if (isinstance(step, bool) or not isinstance(step, int) or step <= self.last_step
                or isinstance(successes, bool) or not isinstance(successes, int) or not 0 <= successes <= 50):
            raise ValueError("selection requires increasing committed steps and a complete50 success count")
        improved = successes > self.best_successes
        if improved:
            self.best_step, self.best_successes = step, successes
            self.non_improving_evaluations = 0
        else:
            self.non_improving_evaluations += 1
        self.last_step = step
        return {"protocol": PROTOCOL, "step": step, "strict_successes": successes,
                "improved": improved, "best_step": self.best_step, "best_successes": self.best_successes,
                "non_improving_evaluations": self.non_improving_evaluations,
                "patience": self.patience, "minimum_updates": self.minimum_updates,
                "should_stop": step >= self.minimum_updates and self.non_improving_evaluations >= self.patience,
                "tie_break": "earlier_checkpoint", "primary_metric": "strict_success_count"}


def summarize_validation(infos, expected_ids, *, step, asynchronous):
    if len(expected_ids) != 50 or len(set(expected_ids)) != 50:
        raise ValueError("checkpoint selection requires the frozen50 validation tasks")
    if [int(i["task_id"]) for i in infos] != list(expected_ids):
        raise ValueError("validation IDs, order or denominator changed")
    invalid = []
    for info in infos:
        if info["strict_success"] != strict_success(info):
            raise ValueError("inconsistent validation strict success")
        if not math.isfinite(float(info["policy_reward"])):
            raise ValueError("nonfinite validation reward")
        overlay = info["training_reward_overlay"]
        if overlay["version"] != "native" or overlay["applied"] or overlay["delta"] != 0:
            raise ValueError("checkpoint selection requires native rewards")
        if asynchronous:
            versions = info.get("sampling_policy_versions")
            if (versions is None or len(versions) != info["assistant_tokens"]
                    or any(v != step for v in versions)):
                raise ValueError("validation sampled stale or mixed policy versions")
        if (not info["reward_valid"] or info.get("reward_unverifiable") or info.get("infrastructure_invalid")
                or info.get("reward", {}).get("sampling_invalid")):
            invalid.append(info["task_id"])
    return {"tasks": 50, "strict_successes": sum(strict_success(i) for i in infos),
            "mean_reward": sum(float(i["policy_reward"]) for i in infos) / 50,
            "mean_steps": sum(i["steps"] for i in infos) / 50,
            "mean_assistant_tokens": sum(i["assistant_tokens"] for i in infos) / 50,
            "invalid_task_ids": sorted(invalid)}


def validate_selection_config(config):
    options = config.get("shopping_checkpoint_selection", {})
    if not options.get("enable", False):
        return
    flash = config.get("shopping_flashreinforce", {})
    if not flash.get("enable") or flash.get("invalid_batch_policy") != "skip_batch":
        raise ValueError("checkpoint selection requires FlashREINFORCE whole-batch admission")
    trainer, data = config["trainer"], config["data"]
    if trainer.get("val_only", False) or trainer.get("resume_mode") != "disable":
        raise ValueError("checkpoint selection requires a fresh training controller")
    interval = options.get("interval", 64)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("checkpoint validation interval must be positive")
    if (trainer.get("test_freq") != interval or trainer.get("save_freq") != interval
            or not trainer.get("val_before_train")):
        raise ValueError("save and validation boundaries must match, including initial validation")
    rollout = config["actor_rollout_ref"]["rollout"]
    val = rollout["val_kwargs"]
    if (data.get("val_batch_size") != 1 or data.get("validation_shuffle", False)
            or not rollout.get("enforce_eager") or rollout.get("enable_prefix_caching", True)
            or val["n"] != 1 or val["do_sample"] or val["temperature"] != 0 or val["top_p"] != 1):
        raise ValueError("checkpoint selection requires ordered serial G1 eager greedy validation")
    SelectionPolicy(options.get("patience", 3), options.get("minimum_updates", 128))


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def transform_selection_fit(source):
    """Use native last-step finalization after an aligned validation boundary."""
    replacements = [
        ('        admission = WholeBatchAdmission.create(self.config, diagnostics_path)\n',
         '        admission = WholeBatchAdmission.create(self.config, diagnostics_path)\n'
         '        from shopping_grpo.training.grpo.checkpoint_selection import TrainingCheckpointSelection\n'
         '        selection = TrainingCheckpointSelection.create(self)\n'),
        ('                    metrics.update(val_metrics)\n',
         '                    metrics.update(val_metrics)\n'
         '                    if selection is not None and selection.should_stop:\n'
         '                        is_last_step = True\n'
         '                        last_val_metrics = val_metrics\n'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError("checkpoint selection native fit insertion point changed")
        source = source.replace(before, after, 1)
    return source


class TrainingCheckpointSelection:
    """Quiesce sampling, validate once serially, pin an adapter, and advise native fit."""

    @classmethod
    def create(cls, trainer):
        if not trainer.config.get("shopping_checkpoint_selection", {}).get("enable", False):
            return None
        selector = cls(trainer)
        trainer._shopping_checkpoint_selector = selector
        trainer._validate = selector.evaluate
        return selector

    def __init__(self, trainer):
        import pyarrow.parquet as pq
        validate_selection_config(trainer.config)
        self.trainer, self.original_validate = trainer, trainer._validate
        self.options = trainer.config.shopping_checkpoint_selection
        self.policy = SelectionPolicy(self.options.patience, self.options.minimum_updates)
        self.should_stop = False
        self.root = Path(os.environ.get("SHOPPING_GRPO_ROOT", Path(__file__).resolve().parents[4]))
        val_files = trainer.config.data.val_files
        paths = [val_files] if isinstance(val_files, str) else list(val_files)
        if len(paths) != 1:
            raise ValueError("checkpoint selection requires one frozen validation parquet")
        self.validation_file = Path(paths[0])
        self.expected = [int(r["extra_info"]["task_id"]) for r in pq.read_table(self.validation_file).to_pylist()]
        frozen = [int(json.loads(s)["task_id"]) for s in (self.root / "data/grpo/validation.jsonl").read_text(encoding="utf-8").splitlines() if s]
        final = {int(json.loads(s)["task_id"]) for s in (self.root / "data/evaluation/tasks.jsonl").read_text(encoding="utf-8").splitlines() if s}
        if len(self.expected) != 50 or set(self.expected) != set(frozen) or final.intersection(self.expected):
            raise ValueError("checkpoint selection cannot consume Final200 or substitute validation tasks")
        self.asynchronous = trainer.config.shopping_flashreinforce.execution == "async"
        self.output = Path(os.environ.get("GRPO_OUTPUT_DIR", trainer.config.trainer.default_local_dir)) / "checkpoint_selection"
        if self.output.exists():
            raise ValueError("checkpoint selection output must be new; preserve previous decisions")
        self.output.mkdir(parents=True)
        self.best_adapter = None

    def restore(self):
        self.trainer._validate = self.original_validate

    def pin_best(self, step):
        parent = Path(self.trainer.config.actor_rollout_ref.model.lora_adapter_path)
        if step == 0:
            adapter = parent
        else:
            adapter = self.output / f"best-step-{step}"
            checkpoint = Path(self.trainer.config.trainer.default_local_dir) / f"global_step_{step}/actor"
            command = [sys.executable, str(self.root / "scripts/export_grpo_adapter.py"),
                       "--checkpoint", str(checkpoint), "--parent-adapter", str(parent),
                       "--output", str(adapter), "--adapter-only"]
            with (self.output / f"export-step-{step}.log").open("wb") as log:
                subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        weights = adapter / "adapter_model.safetensors"
        with weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"step": step, "adapter": str(adapter), "adapter_sha256": digest,
                "native_optimizer_resume_supported": False,
                "preservation": "parent retained" if step == 0 else "durable inference adapter outside native checkpoint rotation"}

    def evaluate(self, *args, **kwargs):
        trainer = self.trainer
        step = int(trainer.global_steps)
        if self.asynchronous and trainer.checkpoint_manager.sampling_version != step:
            raise ValueError("validation weights are not the committed learner version")
        pool = getattr(trainer, "_shopping_flash_pool", None)
        if self.asynchronous and step > 0 and pool is None:
            raise ValueError("async validation cannot proceed without its sampling barrier")
        manager = trainer.async_rollout_manager
        workers, worker_count = manager.agent_loop_workers, trainer.config.actor_rollout_ref.rollout.agent.num_workers
        if not workers:
            raise ValueError("no validation agent worker")
        with pool.pause_dispatch() if pool is not None else nullcontext():
            try:
                manager.agent_loop_workers = workers[:1]
                trainer.config.actor_rollout_ref.rollout.agent.num_workers = 1
                with capture_validation(step) as infos:
                    metrics = self.original_validate(*args, **kwargs)
            finally:
                manager.agent_loop_workers = workers
                trainer.config.actor_rollout_ref.rollout.agent.num_workers = worker_count
            summary = summarize_validation(infos, self.expected, step=step, asynchronous=self.asynchronous)
            decision = self.policy.observe(step, summary["strict_successes"])
            if decision["improved"]:
                self.best_adapter = self.pin_best(step)
                _write(self.output / "best-checkpoint.json", dict(self.best_adapter, **summary, protocol=PROTOCOL))
            self.should_stop = decision["should_stop"]
            record = dict(decision, summary=summary, best_adapter=self.best_adapter,
                          validation_parquet_sha256=hashlib.sha256(self.validation_file.read_bytes()).hexdigest(),
                          training_sampling_quiesced=True, task_records=infos)
            _write(self.output / f"validation-step-{step}.json", record)
            _write(self.output / "state.json", {k: v for k, v in record.items() if k != "task_records"})
            with (self.output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({k: v for k, v in record.items() if k != "task_records"}, ensure_ascii=False) + "\n")
            return {**metrics, "checkpoint_selection/strict_successes": summary["strict_successes"],
                    "checkpoint_selection/best_step": self.policy.best_step,
                    "checkpoint_selection/should_stop": int(self.should_stop)}
