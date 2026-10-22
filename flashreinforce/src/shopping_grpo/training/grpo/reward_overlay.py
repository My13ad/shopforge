"""Training-only necessary-option penalties; native Reward v3 stays authoritative."""
from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
import math
import re
import unicodedata

VERSION = "shopping-spec-penalty-v1"
EVIDENCE_VERSION = "shopping-explicit-options-v1"
MODES = {"native", VERSION}
# None fails closed when an enabled overlay has no native worker lifecycle.
validation_context: ContextVar = ContextVar("shopping_validation_context", default=None)
_SOFT = re.compile(r"最好|优先|尽量|偏好|都可以|均可|不限|随便|或者|或是|"
                   r"\b(?:prefer\w*|ideally|optional|either)\b|if possible", re.I)
_AXES = {"size", "capacity", "color", "dimensions", "net_content", "specification"}


def _normalize(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)).casefold())


def public_instruction(messages):
    """Use only the initial public user message, never gold metadata or system text."""
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not isinstance(messages, (list, tuple)):
        return ""
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(part["text"] for part in content
                                 if isinstance(part, Mapping) and isinstance(part.get("text"), str))
    return ""


def summarize_options(raw_detail, instruction):
    """Retain terminal counts only; do not copy hidden option values to the policy."""
    summary = {"version": EVIDENCE_VERSION, "status": "missing_evidence", "active": False,
               "required_count": 0, "passed_count": 0, "failed_count": 0,
               "unknown_count": 0, "verifiable_count": 0, "native_required_count": 0}
    dimensions = (raw_detail.get("evidence") or {}).get("preference_scoring", {}).get("dimensions", {})
    dimension = dimensions.get("key_options")
    if not isinstance(dimension, Mapping):
        return summary
    results = dimension.get("results")
    count = dimension.get("required_count")
    if not isinstance(results, list) or not isinstance(count, int) or count != len(results):
        raise ValueError("inconsistent native option evidence counts")
    summary["native_required_count"] = count
    if dimension.get("active") is not (count > 0):
        raise ValueError("inconsistent native option active flag")
    if not count:
        summary["status"] = "inactive"
        return summary
    if not instruction:
        summary["status"] = "missing_public_instruction"
        return summary
    if _SOFT.search(instruction):
        summary["status"] = "soft_or_ambiguous_instruction"
        return summary
    text = _normalize(instruction)
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("native option comparison must be an object")
        # Unresolved value fallback does not establish an unambiguous option axis.
        if result.get("comparator") != "option_key_value_v1":
            continue
        requirements = result.get("required")
        if not isinstance(requirements, Mapping) or len(requirements) != 1:
            continue
        axis, requirement = next(iter(requirements.items()))
        if axis not in _AXES or not isinstance(requirement, Mapping):
            continue
        value = str(requirement.get("value") or "")
        if requirement.get("source") != "instruction.instruction_options" or not value:
            continue
        normalized = _normalize(value)
        # ASCII options such as L must not match inside XL or unrelated words.
        pattern = re.escape(normalized)
        if normalized.isascii():
            pattern = r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])"
            mentioned = re.search(pattern, unicodedata.normalize("NFKC", instruction).casefold()) is not None
        else:
            mentioned = normalized in text
        if not mentioned:
            continue
        status = result.get("status")
        if status not in {"pass", "fail", "unverifiable"}:
            raise ValueError("unknown native option comparison status")
        summary["required_count"] += 1
        summary[{"pass": "passed_count", "fail": "failed_count",
                 "unverifiable": "unknown_count"}[status]] += 1
    summary["active"] = summary["required_count"] > 0
    summary["verifiable_count"] = summary["passed_count"] + summary["failed_count"]
    summary["status"] = ("unknown" if summary["unknown_count"] else "verified") if summary["active"] else "not_explicit_or_not_supported"
    return summary


def apply_overlay(reward, detail, *, mode="native", is_validation=None):
    """Return an audit result without changing native validity, success or evidence."""
    if mode not in MODES:
        raise ValueError(f"unsupported training reward overlay: {mode}")
    native = float(reward["total"])
    if not math.isfinite(native):
        raise ValueError("native reward must be finite")
    result = {"version": mode, "applied": False, "reason": "disabled", "delta": 0.,
              "native_reward": native, "training_reward": native, "violation_fraction": 0.}
    if mode == "native":
        return result
    if is_validation is None:
        raise RuntimeError("training reward overlay requires native validation context")
    if is_validation:
        result["reason"] = "native_validation"
        return result
    if reward.get("sampling_invalid"):
        result["reason"] = "invalid_trajectory"
        return result
    if detail.get("reward_type") != "partial_alternative_purchase":
        result["reason"] = "unchanged_native_outcome"
        return result
    gates = detail.get("hard_gates", {})
    if any(gates.get(key, {}).get("status") != "pass" for key in ("category", "budget")):
        result["reason"] = "hard_gate_not_passed"
        return result
    evidence = detail.get("necessary_options", {})
    if evidence.get("version") != EVIDENCE_VERSION or evidence.get("status") != "verified":
        result["reason"] = "unverified_or_inapplicable_options"
        return result
    count, failed = evidence.get("required_count", 0), evidence.get("failed_count", 0)
    if not (0 <= failed <= count) or evidence.get("verifiable_count") != count:
        raise ValueError("inconsistent necessary-option evidence")
    if not failed:
        result["reason"] = "no_necessary_option_failure"
        return result
    score = float(detail["weighted_score"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("match score must be finite and in [0, 1]")
    violation = failed / count
    shaped = -0.20 - 0.20 * violation - 0.10 * (1 - score)
    result.update(applied=True, reason="verified_necessary_option_failure", delta=shaped-native,
                  training_reward=shaped, violation_fraction=violation)
    return result


def install_validation_context_hook():
    """Carry the native validate bit across each coroutine, including padded copies."""
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    original = AgentLoopWorker._run_agent_loop
    if getattr(original, "_shopping_validation_context", False):
        return

    @wraps(original)
    async def run(self, sampling_params, trajectory, *args, **kwargs):
        if not isinstance(trajectory.get("validate"), bool):
            raise ValueError("native trajectory is missing its validate flag")
        token = validation_context.set(trajectory["validate"])
        try:
            return await original(self, sampling_params, trajectory, *args, **kwargs)
        finally:
            validation_context.reset(token)

    run._shopping_validation_context = True
    AgentLoopWorker._run_agent_loop = run
