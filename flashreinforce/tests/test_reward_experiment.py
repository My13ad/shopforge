from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from shopping_grpo.training.grpo.reward_audit import audit_probe
from test_gigpo import info

ROOT = Path(__file__).resolve().parents[1]


def script(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("test_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records():
    result = []
    for task in range(60):
        for index, reward in enumerate((1., .55, .2, -.85)):
            item = info(task=task, reward=reward, action=str(index))
            item["reward"].update(native=reward, sampling_invalid=False)
            item["reward_type"] = "partial_alternative_purchase" if index == 2 else "gold_purchase"
            item["reward_detail"] = {"reward_type": item["reward_type"], "weighted_score": .8,
                "hard_gates": {key: {"status": "pass"} for key in ("category", "budget")},
                "necessary_options": {"version": "shopping-explicit-options-v1", "status": "verified",
                    "required_count": 2, "failed_count": 1, "verifiable_count": 2}}
            result.append({"parent_adapter_sha256": "parent", "shopping": item,
                           "receipt_scope": "post_native_unpad"})
    return result


def test_probe_rescoring_keeps_original_gates_and_does_not_mutate_native_records():
    original = records()
    copy = deepcopy(original)
    report = audit_probe(original, expected_tasks=set(range(60)), parent_sha="parent")
    assert original == copy
    assert report["ready_for_smoke"] and not report["gates_changed"]
    assert report["changed_rollouts"] == 60
    assert report["counterfactual_probe"]["ready_for_main_comparison"]
    assert report["native_probe"]["rollouts"] == 240


def test_missing_rich_evidence_blocks_smoke_instead_of_guessing_from_scores():
    original = records()
    original[2]["shopping"]["reward_detail"] = {}
    report = audit_probe(original, expected_tasks=set(range(60)), parent_sha="parent")
    assert not report["ready_for_smoke"]
    assert report["missing_evidence_task_ids"] == [0]


def test_original_invalid_group_threshold_is_preserved():
    original = records()
    for task in range(4):
        original[task*4]["shopping"]["reward"]["sampling_invalid"] = True
    report = audit_probe(original, expected_tasks=set(range(60)), parent_sha="parent")
    assert not report["ready_for_smoke"]
    assert report["native_probe"]["invalid_task_ids"] == [0, 1, 2, 3]


def test_padding_copies_unknown_parent_or_missing_rows_are_rejected():
    original = records()
    for items, parent in ((original + [original[0]], "parent"), (original[:-1], "parent"),
                           (original, "different")):
        with pytest.raises(ValueError):
            audit_probe(items, expected_tasks=set(range(60)), parent_sha=parent)
    original[0].pop("receipt_scope")
    with pytest.raises(ValueError, match="unpadding"):
        audit_probe(original, expected_tasks=set(range(60)), parent_sha="parent")


def test_restored_candidate_bytes_and_isolation_match_frozen_manifest(monkeypatch):
    module = script(monkeypatch, "prepare_reward_experiment")
    plan = module.prepare()
    assert plan["train_count"] == 953 and plan["validation_count"] == 50
    assert len(set(plan["probe_task_ids"])) == 60
    assert not plan["gpu_queue_created"] and not plan["final_evaluation_authorized"]
    assert plan["code_sha256"]["src/shopping_grpo/training/grpo/reward_overlay.py"]


def test_stage_command_keeps_grpo_parent_and_common_recipe(monkeypatch, tmp_path):
    module = script(monkeypatch, "run_reward_stage")
    plan = module.prepare()
    args = SimpleNamespace(stage="main", model=tmp_path/"model", adapter=tmp_path/"parent",
        output=tmp_path/"result", env_url="http://127.0.0.1:5700", checkpoint_root=tmp_path/"native")
    launch = module.command(args, plan)
    assert launch[launch.index("--method")+1] == "grpo"
    assert launch[launch.index("--profile")+1] == "main"
    assert launch[launch.index("--seed")+1] == "42"
    assert launch[launch.index("--adapter")+1] == str(args.adapter.resolve())
    assert "actor_rollout_ref.actor.calculate_entropy=false" in launch
    assert "actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=true" in launch
    assert "shopping_gigpo.enable=true" not in launch
    assert launch[launch.index("--val-data")+1] == str(ROOT / "data/grpo/validation.parquet")
    assert launch[launch.index("--train-data")+1] == str(ROOT / plan["training_data"])


def test_stage_rejects_changed_plan_before_launching_anything(monkeypatch, tmp_path):
    module = script(monkeypatch, "run_reward_stage")
    plan = module.prepare()
    plan["efficiency_coefficient"] = .05
    with pytest.raises(ValueError, match="current inputs"):
        module.check_prerequisites(SimpleNamespace(), plan)


def test_stage_rejects_wrong_parent_and_missing_probe_receipts(monkeypatch, tmp_path):
    module = script(monkeypatch, "run_reward_stage")
    plan = module.prepare()
    args = SimpleNamespace(stage="smoke", adapter=tmp_path/"parent", output=tmp_path/"result",
        checkpoint_root=None, probe_audit=None, probe_receipts=None)
    monkeypatch.setattr(module, "sha", lambda path: "different")
    with pytest.raises(ValueError, match="original OPD-32"):
        module.check_prerequisites(args, plan)
    monkeypatch.setattr(module, "sha", lambda path: module.PARENT_SHA)
    with pytest.raises(ValueError, match="original receipts"):
        module.check_prerequisites(args, plan)


def test_bundle_rejects_stale_plan_and_failed_checks(monkeypatch, tmp_path):
    module = script(monkeypatch, "build_reward_bundle")
    plan = module.prepare()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    checks = tmp_path / "checks.json"
    checks.write_text(json.dumps({"pytest_exit_code": 1, "gpu_started": False}))
    with pytest.raises(ValueError, match="successful CPU"):
        module.build(tmp_path / "bundle.zip", path, checks)
    checks.write_text(json.dumps({"pytest_exit_code": 0, "gpu_started": False}))
    plan["group_size"] = 8
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="current code/data"):
        module.build(tmp_path / "bundle.zip", path, checks)
