"""Counterfactual checks against the unchanged native reward implementation."""
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from shopping_grpo.training.grpo.adapter.runtime import validate_reward
from shopping_grpo.training.grpo.reward_overlay import (
    VERSION, apply_overlay, install_validation_context_hook, public_instruction,
    summarize_options, validation_context,
)

ROOT = Path(__file__).resolve().parents[1]
NATIVE = next((path for path in (
    ROOT / "environments/ShopSimulator/shop_env",
    ROOT.parent / "sft/environments/ShopSimulator/shop_env",
) if (path / "web_agent_site/engine/reward.py").is_file()), None)


@pytest.fixture
def native(monkeypatch):
    assert NATIVE is not None, "the unchanged native environment source is required"
    monkeypatch.syspath_prepend(str(NATIVE))
    from web_agent_site.engine.reward import evaluate_purchase
    from web_agent_site.engine.reward_features import compile_reward_features
    return evaluate_purchase, compile_reward_features


def case(native, selected="42", instruction="购买黑色42码跑鞋，预算500元", price=300):
    purchase, compile_features = native
    product = {"asin": "111111111111", "title": "跑鞋", "category": "服饰›鞋",
               "pricing": [300], "customization_options": {
                   "尺码": [{"value": "42", "price": 300}, {"value": "38", "price": 300}],
                   "颜色分类": [{"value": "黑色", "price": 300}]}}
    goal = {"asin": product["asin"], "price_upper": 500, **compile_features(
        {"instruction": instruction, "instruction_options": ["黑色", "42"]}, product)}
    result = purchase(product, goal, selected_options={"尺码": selected, "颜色分类": "黑色"}, price=price)
    detail = validate_reward(result.to_dict(), instruction=instruction)
    reward = {"total": result.reward, "terminal_utility": result.reward,
              "sampling_invalid": not result.reward_valid}
    return reward, detail, result.to_dict()


def apply(reward, detail, validation=False):
    return apply_overlay(reward, detail, mode=VERSION, is_validation=validation)


def test_correct_size_above_wrong_size_and_wrong_size_below_eligible_stop(native):
    good, detail, _ = case(native)
    wrong, failed, _ = case(native, selected="38")
    assert apply(good, detail)["training_reward"] == 1.
    new = apply(wrong, failed)
    assert new["applied"] and -.5 <= new["training_reward"] < -.2 < -.15
    assert new["training_reward"] < wrong["total"]
    assert failed["necessary_options"]["required_count"] == 2
    assert failed["necessary_options"]["failed_count"] == 1


@pytest.mark.parametrize("outcome,value", [("gold_purchase", 1.),
    ("valid_alternative_purchase", .55), ("wrong_purchase", -.85),
    ("graceful_stop", -.15), ("early_abstain", -.35), ("repeat_loop", -.65), ("max_steps", -.5)])
def test_other_native_outcomes_are_exactly_preserved(native, outcome, value):
    reward, detail, _ = case(native, selected="38")
    detail["reward_type"] = outcome
    reward["total"] = value
    assert apply(reward, detail)["training_reward"] == value
    assert not apply(reward, detail)["applied"]


def test_budget_and_category_gate_are_not_overridden(native):
    reward, detail, _ = case(native, selected="38", price=800)
    assert detail["hard_gates"]["budget"]["status"] == "fail"
    assert apply(reward, detail)["training_reward"] == -.85
    detail["reward_type"] = "partial_alternative_purchase"
    for gate in ("category", "budget"):
        changed = deepcopy(detail)
        changed["hard_gates"][gate]["status"] = "unverifiable"
        assert not apply(reward, changed)["applied"]


def test_unknown_options_remain_unknown_and_do_not_create_a_new_invalid_filter(native):
    reward, detail, raw = case(native, selected="38")
    options = raw["evidence"]["preference_scoring"]["dimensions"]["key_options"]
    options["results"][0]["status"] = "unverifiable"
    summary = summarize_options(raw, "购买黑色42码跑鞋")
    assert summary["unknown_count"] == 1 and summary["status"] == "unknown"
    detail["necessary_options"] = summary
    before = deepcopy(reward)
    assert not apply(reward, detail)["applied"]
    assert reward == before and not reward["sampling_invalid"]


@pytest.mark.parametrize("instruction", ["最好买黑色42码跑鞋", "黑色优先，42码", "尺寸不限，42码也行"])
def test_preference_language_is_not_silently_hardened(native, instruction):
    reward, detail, _ = case(native, selected="38", instruction=instruction)
    assert not apply(reward, detail)["applied"]
    assert detail["necessary_options"]["status"] == "soft_or_ambiguous_instruction"


def test_inactive_missing_and_unknown_are_distinct(native):
    _, _, raw = case(native)
    missing = summarize_options({}, "购买42码")
    dimension = raw["evidence"]["preference_scoring"]["dimensions"]["key_options"]
    dimension.update(active=False, required_count=0, results=[])
    inactive = summarize_options(raw, "购买42码")
    assert missing["status"] == "missing_evidence" and inactive["status"] == "inactive"


def test_unresolved_fallback_and_unmentioned_gold_options_are_not_hardened(native):
    _, _, raw = case(native)
    summary = summarize_options(raw, "买一双跑鞋")
    assert not summary["active"]
    dimension = raw["evidence"]["preference_scoring"]["dimensions"]["key_options"]
    for result in dimension["results"]:
        result["comparator"] = "exact_option_value_fallback_v1"
    assert not summarize_options(raw, "购买黑色42码跑鞋")["active"]


def test_l_does_not_match_xl(native):
    _, _, raw = case(native)
    dimension = raw["evidence"]["preference_scoring"]["dimensions"]["key_options"]
    result = dimension["results"][0]
    result["required"] = {"size": {"value": "L", "source": "instruction.instruction_options"}}
    dimension.update(required_count=1, results=[result])
    assert not summarize_options(raw, "购买 XL 上衣")["active"]
    assert summarize_options(raw, "购买 L 上衣")["active"]


def test_failure_band_keeps_continuous_signal_and_cannot_be_offset_by_brand(native):
    reward, detail, _ = case(native, selected="38")
    values = []
    for score in (0., .25, .5, .85, 1.):
        detail["weighted_score"] = score
        values.append(apply(reward, detail)["training_reward"])
    assert values == sorted(values) and len(set(values)) == 5
    assert all(-.5 <= value < -.2 for value in values)


def test_validation_and_invalid_trajectories_never_receive_overlay(native):
    reward, detail, _ = case(native, selected="38")
    assert apply(reward, detail, validation=True)["training_reward"] == reward["total"]
    reward["sampling_invalid"] = True
    assert not apply(reward, detail)["applied"]
    with pytest.raises(RuntimeError, match="context"):
        apply_overlay(reward, detail, mode=VERSION)


def test_details_expose_counts_not_hidden_values(native):
    _, detail, _ = case(native, selected="38")
    text = json.dumps(detail["necessary_options"], ensure_ascii=False)
    assert "42" not in text and "黑色" not in text
    assert "required" not in detail and "evidence" not in detail


def test_public_instruction_uses_only_user_content():
    assert public_instruction([{"role": "system", "content": "prefer"},
                               {"role": "user", "content": "购买42码"}]) == "购买42码"
    assert public_instruction(None) == ""


def test_context_hook_is_coroutine_local_reset_on_error_and_idempotent(monkeypatch):
    module = ModuleType("verl.experimental.agent_loop.agent_loop")
    observed = []
    class Worker:
        async def _run_agent_loop(self, sampling, trajectory, **kwargs):
            await asyncio.sleep(0)
            observed.append(validation_context.get())
            if kwargs.get("fail"):
                raise RuntimeError("test failure")
            return validation_context.get()
    module.AgentLoopWorker = Worker
    monkeypatch.setitem(sys.modules, module.__name__, module)
    install_validation_context_hook()
    installed = Worker._run_agent_loop
    install_validation_context_hook()
    assert Worker._run_agent_loop is installed
    async def exercise():
        values = await asyncio.gather(Worker()._run_agent_loop({}, {"validate": True}),
                                      Worker()._run_agent_loop({}, {"validate": False}))
        assert values == [True, False] and validation_context.get() is None
        with pytest.raises(RuntimeError, match="test failure"):
            await Worker()._run_agent_loop({}, {"validate": False}, fail=True)
        assert validation_context.get() is None
    asyncio.run(exercise())


def test_versioned_patch_hash_and_after_unpad_call_match():
    import hashlib
    from test_validation_unpadding import patched_source
    spec = importlib.util.spec_from_file_location("patch_script", ROOT / "scripts/apply_verl_dynamic_sampling_patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = patched_source()
    assert hashlib.sha256(text.encode()).hexdigest() == module.EXPECTED_PATCHED_SHA256
    assert text.index("test_output_gen_batch = unpad_dataproto") < text.index(
        "persist_unpadded_validation_batch(test_output_gen_batch")
