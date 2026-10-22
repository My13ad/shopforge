"""Checkpoint choice uses the complete fixed validation set and committed updates."""
import pytest

from shopping_grpo.training.grpo.checkpoint_selection import (
    SelectionPolicy, summarize_validation,
)


def infos(successes=30, step=0):
    return [dict(task_id=i, done=True, terminal_done=True, terminal_over=True,
                 reward_type="gold_purchase" if i < successes else "repeat_loop",
                 termination_reason="gold_purchase" if i < successes else "repeat_loop",
                 reward_valid=True, infrastructure_invalid=False, reward_unverifiable=False,
                 reward={}, strict_success=i < successes, policy_reward=1.0 if i < successes else -.65,
                 steps=5, assistant_tokens=2, sampling_policy_versions=[step, step],
                 training_reward_overlay={"version": "native", "applied": False, "delta": 0})
            for i in range(50)]


def test_ties_keep_earlier_checkpoint_and_patience_stops_without_a_new_best():
    policy = SelectionPolicy(patience=3, minimum_updates=128)
    assert policy.observe(0, 30)["improved"]
    assert policy.observe(64, 31)["improved"]
    assert not policy.observe(128, 31)["should_stop"]
    assert not policy.observe(192, 30)["should_stop"]
    decision = policy.observe(256, 31)
    assert decision["should_stop"] and decision["best_step"] == 64
    assert decision["best_successes"] == 31 and decision["non_improving_evaluations"] == 3


def test_improvement_resets_patience_and_initial_parent_can_remain_best():
    policy = SelectionPolicy(patience=2, minimum_updates=0)
    policy.observe(0, 30)
    policy.observe(64, 30)
    assert policy.observe(128, 31)["non_improving_evaluations"] == 0
    parent = SelectionPolicy(patience=2, minimum_updates=0)
    parent.observe(0, 30)
    parent.observe(64, 29)
    assert parent.observe(128, 29)["best_step"] == 0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "invented_success", "nonfinite", "wrong_version", "shaping", "reordered"])
def test_incomplete_or_protocol_changed_validation_is_rejected(mutation):
    rows = infos(step=64)
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[-1]["task_id"] = 0
    elif mutation == "invented_success": rows[-1]["strict_success"] = True
    elif mutation == "nonfinite": rows[-1]["policy_reward"] = float("nan")
    elif mutation == "wrong_version": rows[-1]["sampling_policy_versions"] = [63, 64]
    elif mutation == "shaping": rows[-1]["training_reward_overlay"]["applied"] = True
    else: rows.reverse()
    with pytest.raises(ValueError):
        summarize_validation(rows, list(range(50)), step=64, asynchronous=True)


def test_invalid_episode_stays_in_denominator_and_cannot_be_a_success():
    rows = infos()
    rows[0].update(strict_success=False, reward_valid=False, reward_unverifiable=True)
    result = summarize_validation(rows, list(range(50)), step=0, asynchronous=True)
    assert result["tasks"] == 50 and result["strict_successes"] == 29
    assert result["invalid_task_ids"] == [0]


@pytest.mark.parametrize("patience,minimum", [(True, 0), (1.5, 0), (3, -1), (3, True)])
def test_stopping_bounds_are_explicit_integer_counts(patience, minimum):
    with pytest.raises(ValueError):
        SelectionPolicy(patience=patience, minimum_updates=minimum)
