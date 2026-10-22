"""CPU counterfactual reward audit of G4 probes, preserving all existing gates."""
from copy import deepcopy
import math

from shopping_grpo.training.grpo.probe import diagnose_probe
from shopping_grpo.training.grpo.reward_overlay import VERSION, apply_overlay


def audit_probe(records, *, expected_tasks, parent_sha):
    """Rescore the same native rollouts, never optimize on validation/Final tasks."""
    if any(record.get("receipt_scope") != "post_native_unpad" for record in records):
        raise ValueError("reward audit requires probe records after native unpadding")
    original = diagnose_probe(records, expected_tasks=expected_tasks, parent_sha=parent_sha)
    changed = deepcopy(records)
    missing, unknown, applied, deltas, native_scores, training_scores = [], [], [], [], [], []
    for record in changed:
        info = record["shopping"]
        reward = info["reward"]
        detail = info.get("reward_detail") or {}
        if info.get("reward_type") == "partial_alternative_purchase":
            evidence = detail.get("necessary_options", {})
            if not evidence or evidence.get("status") in {"missing_evidence", "missing_public_instruction"}:
                missing.append(int(info["task_id"]))
            if evidence.get("unknown_count", 0):
                unknown.append(int(info["task_id"]))
        # Probes themselves use native validation rewards; this is CPU-only
        # counterfactual scoring of those same trajectories as training samples.
        if abs(float(reward["total"]) - float(reward.get("native", reward["total"]))) > 1e-8:
            raise ValueError("probe was not generated using the native reward")
        overlay = apply_overlay(reward, detail, mode=VERSION, is_validation=False)
        reward["total"] = reward["terminal_utility"] = overlay["training_reward"]
        native_scores.append(overlay["native_reward"])
        training_scores.append(overlay["training_reward"])
        deltas.append(overlay["delta"])
        if overlay["applied"]:
            applied.append(int(info["task_id"]))
    rescored = diagnose_probe(changed, expected_tasks=expected_tasks, parent_sha=parent_sha)
    reasons = []
    if len(expected_tasks) != 60:
        reasons.append("the frozen reward experiment requires probe60")
    if missing:
        reasons.append("partial purchases lack necessary-option evidence")
    if not applied:
        reasons.append("no verified necessary-option failure was observed; inspect coverage")
    if not original["ready_for_main_comparison"]:
        reasons.append("native probe does not pass the unchanged original gates")
    if not rescored["ready_for_main_comparison"]:
        reasons.append("new-reward probe does not pass the unchanged original gates")
    def distribution(values):
        if not values or any(not math.isfinite(v) for v in values):
            raise ValueError("invalid reward values")
        return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}
    return {"schema_version": 1, "overlay_version": VERSION,
            "parent_adapter_sha256": parent_sha, "tasks": len(expected_tasks),
            "rollouts": len(records), "gates_changed": False,
            "native_probe": original, "counterfactual_probe": rescored,
            "changed_rollouts": len(applied), "changed_task_ids": sorted(set(applied)),
            "missing_evidence_task_ids": sorted(set(missing)),
            "unknown_option_task_ids": sorted(set(unknown)),
            "native_rewards": distribution(native_scores),
            "training_rewards": distribution(training_scores), "deltas": distribution(deltas),
            "ready_for_smoke": not reasons, "blocking_reasons": reasons,
            "scope": "CPU counterfactual scoring only; not proof of training benefit"}
