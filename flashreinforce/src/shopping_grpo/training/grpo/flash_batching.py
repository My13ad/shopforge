"""Bounded whole-batch admission before FlashREINFORCE learning work.

The source-pinned fit hook is process-local; shared veRL files remain intact.
Native one-based in-flight steps advance only after an admitted update.
"""
from __future__ import annotations

import ast
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path

from shopping_grpo.training.grpo.dynamic_sampling import (
    append_training_diagnostic, build_rollout_diagnostics, extract_shopping_group_signals,
)

PROTOCOL = "flash-whole-batch-native-v1"
VERL_SOURCE_SHA256 = "cc4544446cbb878eccfe6a1212ce1c3dfe31d05062088df4e36db850a40fad53"
POLICIES = ("error", "skip_batch")


def validate_admission_config(config):
    options = config.get("shopping_flashreinforce", {})
    policy = options.get("invalid_batch_policy", "error")
    if policy not in POLICIES:
        raise ValueError(f"unknown FlashREINFORCE invalid_batch_policy: {policy}")
    if policy != "skip_batch":
        return
    for key in ("max_candidate_batches", "max_consecutive_skipped_batches"):
        value = options.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"shopping_flashreinforce.{key} must be a positive integer")
    target = config["trainer"].get("total_training_steps")
    if target is not None and int(options["max_candidate_batches"]) < int(target):
        raise ValueError("max_candidate_batches cannot be smaller than the optimizer update budget")
    # Native resume derives epochs from optimizer steps rather than consumed
    # candidates. Do not claim support for resuming that inconsistent position.
    if config["trainer"].get("resume_mode") != "disable":
        raise ValueError("whole-batch admission currently requires trainer.resume_mode=disable")
    if config["actor_rollout_ref"]["rollout"].get("skip", {}).get("enable", False):
        raise ValueError("whole-batch admission requires fresh rollouts without rollout.skip")


class WholeBatchAdmission:
    def __init__(self, config, diagnostics_path):
        validate_admission_config(config)
        if not diagnostics_path:
            raise ValueError("whole-batch admission requires persistent training diagnostics")
        self.config = config
        self.options = dict(config["shopping_flashreinforce"])
        self.path = diagnostics_path
        self.candidates = self.skipped = self.consecutive = 0
        self.seen = set()
        self.feedback_counts = {stage: {"total": 0, "hard_lane": 0, "hard_member": 0}
                                for stage in ("candidate", "admitted", "skipped")}

    @classmethod
    def create(cls, config, diagnostics_path):
        options = config.get("shopping_flashreinforce", {})
        if options.get("enable", False) and options.get("invalid_batch_policy", "error") == "skip_batch":
            return cls(config, diagnostics_path)
        return None

    def check_budget(self, current_step):
        if self.candidates >= self.options["max_candidate_batches"]:
            raise RuntimeError(f"FlashREINFORCE candidate batch limit reached before update {current_step}")

    def consider(self, batch, current_step):
        self.check_budget(current_step)
        infos = list(batch.non_tensor_batch["shopping"])
        uids = list(batch.non_tensor_batch["uid"])
        expected = int(self.config["data"]["train_batch_size"])
        mask = batch.batch["response_mask"]
        if mask.ndim != 2 or len(mask) != expected or len(infos) != expected or len(uids) != expected:
            raise ValueError("incomplete or misaligned FlashREINFORCE candidate batch")
        if len(set(uids)) != expected or len({info["task_id"] for info in infos}) != expected:
            raise ValueError("FlashREINFORCE candidate batch requires distinct uids and tasks")
        if self.seen.intersection(uids):
            raise ValueError("a FlashREINFORCE candidate rollout cannot be reused")
        _, _, invalid, reasons = extract_shopping_group_signals(infos)
        self.seen.update(uids)
        self.candidates += 1
        admitted = not any(invalid)
        self.skipped += int(not admitted)
        self.consecutive = 0 if admitted else self.consecutive + 1
        record = {
            "protocol": PROTOCOL, "candidate_batch": self.candidates,
            "target_optimizer_step": current_step, "decision": "admit" if admitted else "skip_batch",
            "batch_size": expected, "invalid_trajectories": sum(invalid), "invalid_reasons": reasons,
            "skipped_batches_total": self.skipped, "consecutive_skipped_batches": self.consecutive,
            "rollouts": build_rollout_diagnostics(uids, infos),
        }
        if self.config.get("shopping_feedback_sampling", {}).get("enable", False):
            choices = [info.get("feedback_sampling", {}) for info in infos]
            for uid, info, choice in zip(uids, infos, choices, strict=True):
                if choice.get("uid") != uid or choice.get("task_id") != info["task_id"] or choice.get("lane") not in ("hard", "background"):
                    raise ValueError("feedback admission lost sampler provenance")
            for stage in ("candidate", "admitted" if admitted else "skipped"):
                count = self.feedback_counts[stage]
                count["total"] += expected
                count["hard_lane"] += sum(c["lane"] == "hard" for c in choices)
                count["hard_member"] += sum(c["hard_member"] for c in choices)
            record["feedback_sampling_counts"] = {k: dict(v) for k, v in self.feedback_counts.items()}
        append_training_diagnostic(self.path, "flash_batch_admission", current_step - 1, **record)
        print("SHOPPING_FLASH_BATCH_ADMISSION " + json.dumps({k: v for k, v in record.items()
              if k != "rollouts"}, sort_keys=True))
        if not admitted and self.consecutive >= self.options["max_consecutive_skipped_batches"]:
            raise RuntimeError("FlashREINFORCE consecutive skipped batch limit reached")
        if not admitted and self.candidates >= self.options["max_candidate_batches"]:
            raise RuntimeError("FlashREINFORCE candidate batch limit reached")
        return admitted

    def metrics(self):
        return {
            "flashreinforce/candidate_batches_total": self.candidates,
            "flashreinforce/skipped_batches_total": self.skipped,
            "flashreinforce/consecutive_skipped_batches": self.consecutive,
        }

    def save_checkpoint(self, step):
        destination = Path(self.config["trainer"]["default_local_dir"]) / f"global_step_{step}"
        if not destination.is_dir():
            raise RuntimeError("missing native checkpoint for admission audit")
        state = {
            "protocol": PROTOCOL, "optimizer_updates": step, "candidate_batches": self.candidates,
            "skipped_batches": self.skipped, "consecutive_skipped_batches": self.consecutive,
            "seen_rollout_uids": sorted(self.seen), "settings": self.options,
            "native_resume_supported": False,
        }
        temporary = destination / "flash_batch_admission.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination / "flash_batch_admission.json")


def prepare_fit(source, namespace, *, asynchronous=False):
    """Compile the native fit with checked, narrow insertion points for admission."""
    if hashlib.sha256(source.encode()).hexdigest() != VERL_SOURCE_SHA256:
        raise RuntimeError("FlashREINFORCE admission requires the exact pinned veRL V4 source")
    replacements = [
        ('        diagnostics_path = os.environ.get("SHOPPING_GRPO_DIAGNOSTICS_PATH")\n',
         '        diagnostics_path = os.environ.get("SHOPPING_GRPO_DIAGNOSTICS_PATH")\n'
         '        from shopping_grpo.training.grpo.flash_batching import WholeBatchAdmission\n'
         '        admission = WholeBatchAdmission.create(self.config, diagnostics_path)\n'),
        ('                optimizer_updated = False\n',
         '                optimizer_updated = False\n'
         '                if admission is not None:\n'
         '                    admission.check_budget(current_step)\n'),
        ('                        if not dynamic_sampling_enabled:\n'
         '                            self.checkpoint_manager.sleep_replicas()\n',
         '                        if not dynamic_sampling_enabled and admission is None:\n'
         '                            self.checkpoint_manager.sleep_replicas()\n'),
        ('                    if dynamic_sampling_enabled:\n'
         '                        seq_rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()\n',
         '                    if admission is not None:\n'
         '                        if not admission.consider(batch, current_step):\n'
         '                            self._stop_profiling(curr_step_profile)\n'
         '                            prev_step_profile = False\n'
         '                            continue\n'
         '                        metrics.update(admission.metrics())\n'
         '                        self.checkpoint_manager.sleep_replicas()\n\n'
         '                    if dynamic_sampling_enabled:\n'
         '                        seq_rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()\n'),
        ('                                self._save_checkpoint()\n',
         '                                self._save_checkpoint()\n'
         '                                if admission is not None:\n'
         '                                    admission.save_checkpoint(self.global_steps)\n'),
        ('        # Ensure dump executor is shut down when training loop ends without reaching is_last_step\n'
         '        self._shutdown_dump_executor()\n',
         '        # Ensure dump executor is shut down when training loop ends without reaching is_last_step\n'
         '        self._shutdown_dump_executor()\n'
         '        if admission is not None:\n'
         '            raise RuntimeError("FlashREINFORCE dataloader exhausted before optimizer update budget")\n'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError("FlashREINFORCE native fit insertion point changed")
        source = source.replace(before, after, 1)
    from shopping_grpo.training.grpo.checkpoint_selection import transform_selection_fit
    source = transform_selection_fit(source)
    if asynchronous:
        from shopping_grpo.training.grpo.async_fit import transform_fit
        source = transform_fit(source)
    tree = ast.parse(source)
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    fit = next(n for n in trainer.body if getattr(n, "name", None) == "fit")
    compiled = compile(ast.Module(body=[fit], type_ignores=[]), f"<{PROTOCOL}>", "exec")
    local = {}
    exec(compiled, namespace, local)
    return local["fit"]


def install_batch_admission(trainer_class):
    original = trainer_class.fit
    if getattr(original, "_shopping_flash_batching", False):
        return
    prepared = {}

    @wraps(original)
    def fit(self):
        options = self.config.get("shopping_flashreinforce", {})
        if not options.get("enable", False) or options.get("invalid_batch_policy", "error") != "skip_batch":
            return original(self)
        from shopping_grpo.training.grpo.flashreinforce import validate_flashreinforce_config
        validate_flashreinforce_config(self.config)
        asynchronous = options.get("execution", "sync") == "async"
        if asynchronous not in prepared:
            source = Path(inspect.getsourcefile(original)).read_text(encoding="utf-8")
            prepared[asynchronous] = prepare_fit(source, original.__globals__, asynchronous=asynchronous)
        try:
            return prepared[asynchronous](self)
        finally:
            selector = getattr(self, "_shopping_checkpoint_selector", None)
            if selector is not None:
                selector.restore()
            pool = getattr(self, "_shopping_flash_pool", None)
            if asynchronous and pool is not None:
                pool.close()
            feedback = getattr(self, "_shopping_feedback_source", None)
            if feedback is not None:
                feedback.close()

    fit._shopping_flash_batching = True
    trainer_class.fit = fit
