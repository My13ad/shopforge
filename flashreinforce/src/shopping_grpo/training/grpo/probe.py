"""No-gradient probe receipts and training-only state-group diagnostics."""
from __future__ import annotations
import json
import os
import uuid
from pathlib import Path

from shopping_grpo.training.grpo.gigpo import state_credit


def persist_probe_rollout(info: dict) -> None:
    destination = os.environ.get("SHOPPING_GIGPO_PROBE_DIR")
    if not destination:
        return
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{uuid.uuid4().hex}.json"
    record = {"schema_version": 1, "parent_adapter_sha256": os.environ.get(
        "SHOPPING_PARENT_ADAPTER_SHA256", ""), "shopping": info,
        "receipt_scope": "post_native_unpad"}
    path.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def diagnose_probe(records: list[dict], *, expected_tasks: set[int], parent_sha: str,
                   group_size: int = 4, gamma: float = 1.0) -> dict:
    by_task = {}
    for record in records:
        if not parent_sha or record.get("parent_adapter_sha256") != parent_sha:
            raise ValueError("probe contains a different or unidentified parent policy")
        info = record["shopping"]
        task = int(info["task_id"])
        if task not in expected_tasks:
            raise ValueError("probe task is outside the frozen training subset")
        by_task.setdefault(task, []).append(info)
    if set(by_task) != expected_tasks or any(len(g) != group_size for g in by_task.values()):
        raise ValueError("probe needs exactly G completed rollouts for every selected task")
    valid, invalid_tasks, varying_tasks = [], [], []
    for task, infos in by_task.items():
        if any(info.get("infrastructure_invalid") or info.get("reward_unverifiable") or
               info["reward"].get("sampling_invalid", False) for info in infos):
            invalid_tasks.append(task)
            continue
        valid.extend(infos)
        rewards = [float(info["reward"]["total"]) for info in infos]
        if max(rewards) - min(rewards) > 1e-8:
            varying_tasks.append(task)

    def coverage(infos):
        if not infos:
            return {}
        return state_credit([str(i["task_id"]) for i in infos], infos,
                            [i["reward"]["total"] for i in infos], gamma=gamma,
                            expected_group_size=group_size)[1]

    accepted_coverage = coverage([i for i in valid if int(i["task_id"]) in varying_tasks])
    # Operational gates to avoid a long run with only redundant initial-state credit.
    ready = (len(expected_tasks) >= 32 and len(varying_tasks) >= 8 and
             len(invalid_tasks) / len(expected_tasks) <= .05 and
             accepted_coverage.get("gigpo/effective_noninitial_turns", 0) >= 16 and
             accepted_coverage.get("gigpo/effective_noninitial_turn_fraction", 0) >= .1)
    return {"schema_version": 1, "parent_adapter_sha256": parent_sha,
            "group_size": group_size, "tasks": len(expected_tasks), "rollouts": len(records),
            "invalid_task_ids": sorted(invalid_tasks), "reward_varying_task_ids": sorted(varying_tasks),
            "all_valid_coverage": coverage(valid),
            "dynamic_sampling_eligible_coverage": accepted_coverage,
            "ready_for_main_comparison": ready,
            "decision": "two_arm_comparison" if ready else "inspect_probe_before_training"}
