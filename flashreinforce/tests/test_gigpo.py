from copy import deepcopy
from types import SimpleNamespace
import random
import pytest
import torch

from shopping_grpo.training.grpo.gigpo import (
    SCHEMA_VERSION, apply_gigpo_advantages, finalize_turns, public_state_key,
    record_generation, state_credit, validate_gigpo_config,
)
from shopping_grpo.training.grpo.adapter.runtime import make_runtime_state
from shopping_grpo.training.grpo.probe import diagnose_probe


def info(task=1, keys=("initial", "shared"), reward=1.0, action="a"):
    return {"task_id": task, "reward": {"total": reward}, "gigpo": {
        "schema_version": SCHEMA_VERSION, "response_length": 7,
        "turns": [{"start": 0, "end": 2, "state_key": keys[0], "action_key": action,
                   "turn_index": 0},
                  {"start": 4, "end": 6, "state_key": keys[1], "action_key": action,
                   "turn_index": 1}]}}


def batch(infos=None):
    infos = infos or [info(reward=r, action=str(i)) for i, r in enumerate([1., .5, 0., -.5])]
    rewards = torch.zeros(len(infos), 8)
    rewards[:, 6] = torch.tensor([i["reward"]["total"] for i in infos])
    mask = torch.tensor([[1, 1, 0, 0, 1, 1, 0, 0]] * len(infos))
    centered = rewards.sum(-1) - rewards.sum(-1).mean()
    return SimpleNamespace(batch={"response_mask": mask,
                                 "advantages": centered[:, None] * mask,
                                 "token_level_rewards": rewards},
                           non_tensor_batch={"uid": ["u"] * len(infos), "shopping": infos})


def test_paper_additive_formula_and_masking():
    data = batch()
    before = deepcopy(data.batch)
    metrics = apply_gigpo_advantages(data)
    assert torch.equal(data.batch["advantages"], 2 * before["advantages"])
    assert metrics["gigpo/effective_noninitial_turns"] == 4
    assert torch.equal(data.batch["response_mask"], before["response_mask"])
    assert torch.equal(data.batch["token_level_rewards"], before["token_level_rewards"])
    assert torch.equal(data.batch["advantages"][:, [2, 3, 6, 7]], torch.zeros(4, 4))


def test_omega_zero_recovers_baseline_bitwise():
    data = batch()
    before = data.batch["advantages"].clone()
    apply_gigpo_advantages(data, omega=0)
    assert torch.equal(data.batch["advantages"], before)


def test_real_step_baseline_differs_from_episode_baseline():
    infos = [info(keys=("i", "left"), reward=r) for r in (1., 0.)]
    infos += [info(keys=("i", "right"), reward=r) for r in (.4, .2)]
    data = batch(infos)
    original = data.batch["advantages"].clone()
    apply_gigpo_advantages(data)
    assert torch.allclose(data.batch["advantages"][:, 4] - original[:, 4],
                          torch.tensor([.5, -.5, .1, -.1]))


def test_singletons_add_no_local_credit():
    infos = [info(keys=("i", f"s{i}"), reward=r) for i, r in enumerate((1., .5, 0., -.5))]
    data = batch(infos)
    before = data.batch["advantages"].clone()
    metrics = apply_gigpo_advantages(data)
    assert torch.equal(data.batch["advantages"][:, 4:6], before[:, 4:6])
    assert metrics["gigpo/effective_noninitial_turns"] == 0


def test_zero_variance_group_has_no_credit():
    infos = [info(reward=1., action=str(i)) for i in range(4)]
    values, metrics = state_credit(["u"] * 4, infos, [1.] * 4)
    assert values == [[0., 0.]] * 4
    assert metrics["gigpo/effective_state_groups"] == 0


def test_discounted_terminal_return_matches_hand_calculation():
    infos = [info(reward=1.), info(reward=0.)]
    values, _ = state_credit(["u"] * 2, infos, [1., 0.], gamma=.95)
    assert values[0] == pytest.approx([.475, .5])
    assert values[1] == pytest.approx([-.475, -.5])


def test_groups_isolate_uid_even_for_same_task():
    infos = [info(reward=r) for r in (1., 0., .4, .2)]
    values, _ = state_credit(["u", "u", "v", "v"], infos, [1., 0., .4, .2])
    assert [v[1] for v in values] == pytest.approx([.5, -.5, .1, -.1])


def test_permutation_equivariance_after_batch_balancing():
    infos = [info(reward=r) for r in (1., .5, 0., -.5)]
    values, _ = state_credit(["u"] * 4, infos, [i["reward"]["total"] for i in infos])
    order = [2, 0, 3, 1]
    shuffled = [infos[i] for i in order]
    actual, _ = state_credit(["u"] * 4, shuffled, [i["reward"]["total"] for i in shuffled])
    assert actual == [values[i] for i in order]


def test_duplicate_visit_does_not_bias_state_baseline():
    infos = [info(keys=("repeat", "repeat"), reward=1.), info(reward=0., keys=("repeat", "x"))]
    values, _ = state_credit(["u"] * 2, infos, [1., 0.])
    assert values == [[.5, 0.], [-.5, 0.]]


def test_public_key_excludes_hidden_data_and_preserves_options_price():
    state = make_runtime_state(1, 35)
    state["latest_observation"] = 'ASIN: B123 price: 30 selected: red, XL'
    key = public_state_key(state)
    state.update(gold_asin="SECRET", terminal_result={"gold": "hidden"}, reward_detail={"score": 9},
                 latest_observation_raw="unprojected SECRET")
    assert public_state_key(state) == key
    for text in ('ASIN: B123 price: 31 selected: red, XL',
                 'ASIN: B123 price: 30 selected: blue, XL'):
        state["latest_observation"] = text
        assert public_state_key(state) != key


def test_public_guard_state_prevents_aliasing():
    state = make_runtime_state(1, 35)
    state["latest_observation"] = "page"
    key = public_state_key(state)
    state["consecutive_guard_rejections"] = 1
    assert public_state_key(state) != key


def test_same_public_page_and_horizon_can_match_after_different_paths():
    """Reproduce probe trajectories that opened the same product by different searches."""
    left = make_runtime_state(1, 35)
    left.update(latest_observation="ASIN: B123 price: 30 selected: red, XL",
                action_attempt_count=2,
                recent_action_signatures=[("search_products", '{"query":"red shirt"}', "results-a"),
                                          ("open_product", '{"asin":"B123"}', "results-a")])
    right = deepcopy(left)
    right["recent_action_signatures"] = [
        ("search_products", '{"query":"XL shirt"}', "results-b"),
        ("open_product", '{"asin":"B123"}', "results-b")]
    assert public_state_key(left) == public_state_key(right)
    right["action_attempt_count"] += 1
    assert public_state_key(left) != public_state_key(right)


def test_recording_offsets_include_tool_tokens_and_final_truncation():
    state = {}
    agent = SimpleNamespace(response_mask=[1, 1, 0, 0, 1, 1, 1], response_ids=[9, 8, 7])
    record_generation(state, agent, start=4, key="s")
    assert state["gigpo_turns"][0]["start"] == 4
    assert finalize_turns(state, 6)["turns"][0]["end"] == 6
    assert finalize_turns(state, 4)["turns"] == []


@pytest.mark.parametrize("corruption", ["missing", "schema", "tool", "overlap", "uncovered", "invalid", "mixed_task", "nan", "reward"])
def test_invalid_metadata_fails_before_mutating_batch(corruption):
    data = batch()
    if corruption == "missing":
        data.non_tensor_batch["shopping"][0].pop("gigpo")
    elif corruption == "schema":
        data.non_tensor_batch["shopping"][0]["gigpo"]["schema_version"] = 1
    elif corruption == "tool":
        data.non_tensor_batch["shopping"][0]["gigpo"]["turns"][0]["end"] = 3
    elif corruption == "overlap":
        data.non_tensor_batch["shopping"][0]["gigpo"]["turns"][1]["start"] = 1
    elif corruption == "uncovered":
        data.non_tensor_batch["shopping"][0]["gigpo"]["turns"][1]["end"] = 5
    elif corruption == "invalid":
        data.non_tensor_batch["shopping"][0]["infrastructure_invalid"] = True
    elif corruption == "mixed_task":
        data.non_tensor_batch["shopping"][0]["task_id"] = 2
    elif corruption == "nan":
        data.batch["token_level_rewards"][0, 6] = float("nan")
    else:
        data.batch["token_level_rewards"][0, 6] = .9
    before = data.batch["advantages"].clone()
    with pytest.raises(ValueError):
        apply_gigpo_advantages(data)
    assert torch.equal(data.batch["advantages"], before)


def test_incomplete_group_is_rejected():
    data = batch()
    data.non_tensor_batch["uid"] = ["u", "u", "u", "v"]
    with pytest.raises(ValueError, match="incomplete"):
        apply_gigpo_advantages(data)


def test_probe_does_not_treat_initial_only_credit_as_ready():
    records = [{"parent_adapter_sha256": "sha", "shopping": info(task=t,
                keys=("initial", f"s{j}"), reward=float(j == 0))}
               for t in range(32) for j in range(4)]
    report = diagnose_probe(records, expected_tasks=set(range(32)), parent_sha="sha")
    assert len(report["reward_varying_task_ids"]) == 32
    assert report["ready_for_main_comparison"] is False


def test_probe_with_noninitial_coverage_can_pass():
    records = [{"parent_adapter_sha256": "sha", "shopping": info(task=t,
                reward=float(j == 0), action=str(j))}
               for t in range(32) for j in range(4)]
    assert diagnose_probe(records, expected_tasks=set(range(32)), parent_sha="sha")["ready_for_main_comparison"]


@pytest.mark.parametrize("failure", ["parent", "extra", "missing"])
def test_probe_rejects_mixed_policy_or_incomplete_data(failure):
    records = [{"parent_adapter_sha256": "sha", "shopping": info(task=1)} for _ in range(4)]
    if failure == "parent":
        records[0]["parent_adapter_sha256"] = "other"
    elif failure == "extra":
        records[0]["shopping"]["task_id"] = 2
    else:
        records.pop()
    with pytest.raises(ValueError):
        diagnose_probe(records, expected_tasks={1}, parent_sha="sha")


def test_trace_and_gigpo_cannot_be_combined():
    config = {"shopping_gigpo": {"enable": True, "omega": 1, "gamma": 1, "reward_tolerance": 1e-8},
              "shopping_trace": {"enable": True}}
    with pytest.raises(ValueError, match="separately"):
        validate_gigpo_config(config)
