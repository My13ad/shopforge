"""GiGPO public-state adaptation (arXiv:2505.10978v3, Eq. 8).

No extra model or rollout. Exact public-state groups and one first visit per
rollout are conservative adaptations of the paper's all-occurrences grouping.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

SCHEMA_VERSION = 2


def fingerprint(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def public_state_key(state: Mapping) -> str:
    """Match exact visible pages at the same public horizon and guard count.

    Exact text retains ASIN, price and selected options. Version 2 excludes the
    recent action signatures: distinct paths can reach a comparable public page.
    This is an approximate anchor baseline, not a Markov state or causal value:
    conversation history and the environment's progress tracker may still differ.
    """
    observation = state.get("latest_observation")
    if not isinstance(observation, str) or not observation:
        raise ValueError("GiGPO needs a nonempty model-visible observation")
    return fingerprint({
        "schema": SCHEMA_VERSION, "task_id": int(state["task_id"]),
        "observation": observation, "max_steps": int(state["max_steps"]),
        "action_attempts": int(state["action_attempt_count"]),
        "guard_rejections": int(state["consecutive_guard_rejections"]),
    })


def record_generation(state: dict, agent_data, *, start: int, key: str) -> None:
    """Offsets refer to the response, including zero-mask tool observations."""
    end = len(agent_data.response_mask)
    if end <= start:
        return
    if not all(agent_data.response_mask[start:end]):
        raise ValueError("generation span contains non-assistant tokens")
    turns = state.setdefault("gigpo_turns", [])
    turns.append({"start": start, "end": end, "state_key": key,
                  "action_key": fingerprint(agent_data.response_ids),
                  "turn_index": len(turns)})


def finalize_turns(state: Mapping, response_length: int) -> dict:
    """Mirror ToolAgentLoop's right truncation without altering any tokens."""
    turns = []
    for turn in state.get("gigpo_turns", []):
        if turn["start"] >= response_length:
            break
        turns.append({**turn, "end": min(turn["end"], response_length)})
    return {"schema_version": SCHEMA_VERSION, "response_length": response_length,
            "turns": turns}


def _validate_options(omega: float, gamma: float, tolerance: float) -> None:
    if not math.isfinite(omega) or omega < 0:
        raise ValueError("omega must be finite and nonnegative")
    if not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")


def state_credit(uids: Sequence, shopping_infos: Sequence, rewards: Sequence[float],
                 *, gamma: float = 1.0, tolerance: float = 1e-8,
                 expected_group_size: int | None = None) -> tuple[list, dict]:
    """Compute first-visit state advantages and coverage on CPU, with F_norm=1.

    Terminal-only return is gamma**(T-1-t) * R. Singleton states keep the original
    episode advantage. Repeated visits in one rollout cannot multiply its weight.
    """
    _validate_options(1.0, gamma, tolerance)
    if not (len(uids) == len(shopping_infos) == len(rewards)) or not len(uids):
        raise ValueError("nonempty uids, shopping_infos and rewards must align")
    groups, anchors = defaultdict(list), defaultdict(list)
    values = []
    turn_count = noninitial = 0
    for row, (uid, info, reward) in enumerate(zip(uids, shopping_infos, rewards, strict=True)):
        hash(uid)
        groups[uid].append(row)
        if not math.isfinite(float(reward)):
            raise ValueError("nonfinite terminal reward")
        if not isinstance(info, Mapping) or "task_id" not in info:
            raise ValueError("missing shopping task identity")
        metadata = info.get("gigpo", {})
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("missing or incompatible GiGPO turn metadata")
        turns = metadata.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError("GiGPO needs a generated assistant turn")
        if info.get("infrastructure_invalid") or info.get("reward_unverifiable") or (
            info.get("reward", {}).get("sampling_invalid", False)
        ):
            raise ValueError("invalid rollout must be removed as an entire prompt group")
        values.append([0.0] * len(turns))
        seen = set()
        previous_end = 0
        for turn_index, turn in enumerate(turns):
            start, end = turn["start"], turn["end"]
            if (type(start) is not int or type(end) is not int or
                    not previous_end <= start < end <= metadata["response_length"]):
                raise ValueError("turn spans must be ordered, nonoverlapping and in bounds")
            if turn.get("turn_index") != turn_index:
                raise ValueError("turn indices must be sequential")
            previous_end = end
            key, action = turn["state_key"], turn["action_key"]
            if not isinstance(key, str) or not key or not isinstance(action, str) or not action:
                raise ValueError("state/action fingerprints must be nonempty strings")
            turn_count += 1
            noninitial += int(turn_index > 0)
            if key in seen:
                continue
            seen.add(key)
            downstream = gamma ** (len(turns) - 1 - turn_index) * float(reward)
            anchors[(uid, key)].append((row, turn_index, downstream, action))
    for rows in groups.values():
        if len({int(shopping_infos[row]["task_id"]) for row in rows}) != 1:
            raise ValueError("a prompt UID contains multiple tasks")
        if len(rows) < 2 or (expected_group_size is not None and len(rows) != expected_group_size):
            raise ValueError("incomplete on-policy rollout group")
    matched = effective = effective_noninitial = divergent = 0
    same_action = matched_groups = effective_groups = 0
    for occurrences in anchors.values():
        if len(occurrences) < 2:
            continue
        matched_groups += 1
        matched += len(occurrences)
        if len({item[3] for item in occurrences}) < 2:
            same_action += 1
        else:
            divergent += 1
        downstreams = [item[2] for item in occurrences]
        if max(downstreams) - min(downstreams) <= tolerance:
            continue
        effective_groups += 1
        mean = sum(downstreams) / len(downstreams)
        for row, turn_index, downstream, _ in occurrences:
            values[row][turn_index] = downstream - mean
            effective += 1
            effective_noninitial += int(turn_index > 0)
    return values, {
        "gigpo/turns": turn_count, "gigpo/noninitial_turns": noninitial,
        "gigpo/matched_state_groups": matched_groups,
        "gigpo/effective_state_groups": effective_groups,
        "gigpo/divergent_action_groups": divergent, "gigpo/same_action_groups": same_action,
        "gigpo/matched_turn_fraction": matched / max(turn_count, 1),
        "gigpo/effective_turn_fraction": effective / max(turn_count, 1),
        "gigpo/effective_noninitial_turn_fraction": effective_noninitial / max(noninitial, 1),
        "gigpo/effective_noninitial_turns": effective_noninitial,
        "gigpo/state_adv_abs_mean": sum(abs(x) for row in values for x in row) / max(turn_count, 1),
    }


def apply_gigpo_advantages(batch, *, omega: float = 1.0, gamma: float = 1.0,
                           tolerance: float = 1e-8, expected_group_size: int = 4) -> dict:
    """Only advantages change; rewards, masks, old logps and actor loss stay fixed."""
    import torch

    _validate_options(omega, gamma, tolerance)
    advantages, mask = batch.batch["advantages"], batch.batch["response_mask"]
    if advantages.shape != mask.shape or advantages.ndim != 2:
        raise ValueError("advantages and response_mask need identical 2D shape")
    if not torch.isfinite(advantages).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("invalid advantages or response_mask")
    infos, uids = list(batch.non_tensor_batch["shopping"]), list(batch.non_tensor_batch["uid"])
    if len(infos) != advantages.shape[0]:
        raise ValueError("shopping metadata and tensor rows differ")
    rewards = batch.batch["token_level_rewards"].sum(-1).detach().cpu().tolist()
    values, metrics = state_credit(uids, infos, rewards, gamma=gamma, tolerance=tolerance,
                                   expected_group_size=expected_group_size)
    delta = torch.zeros_like(advantages)
    for row, info in enumerate(infos):
        if not math.isclose(float(info["reward"]["total"]), rewards[row], rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("GiGPO terminal-only mode needs unmodified Reward v3")
        coverage = torch.zeros_like(mask[row], dtype=torch.bool)
        for turn, value in zip(info["gigpo"]["turns"], values[row], strict=True):
            start, end = turn["start"], turn["end"]
            if end > mask.shape[1] or not torch.all(mask[row, start:end] == 1):
                raise ValueError("turn span includes observation or padding tokens")
            coverage[start:end] = True
            delta[row, start:end] = omega * value
        if not torch.equal(coverage, mask[row].bool()):
            raise ValueError("turn metadata does not cover every assistant token")
    # All validation finishes before mutation; omega=0 keeps baseline bitwise.
    batch.batch["advantages"] = advantages + delta
    metrics["gigpo/additive_adv_abs_mean"] = float(delta.abs().sum() / mask.sum().clamp_min(1))
    return metrics


def validate_gigpo_config(config) -> None:
    method = config.get("shopping_gigpo", {})
    if not method.get("enable", False):
        return
    _validate_options(float(method["omega"]), float(method["gamma"]),
                      float(method["reward_tolerance"]))
    if config.get("shopping_trace", {}).get("enable", False):
        raise ValueError("GiGPO and TRACE must be evaluated separately")
    algorithm = config["algorithm"]
    if (str(algorithm["adv_estimator"]) != "grpo" or
            algorithm.get("norm_adv_by_std_in_grpo", True) or algorithm.get("use_kl_in_reward", False)):
        raise ValueError("GiGPO needs unnormalized terminal-only GRPO advantages")
    if int(config["actor_rollout_ref"]["rollout"]["n"]) < 2:
        raise ValueError("GiGPO needs at least two on-policy rollouts per prompt")
