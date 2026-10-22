import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from shopping_grpo.training.grpo.diagnostics import (
    annotate_rollout_statistics, compare_tasks, persist_unpadded_validation_batch,
    run_identity_audit, samples_equal,
    strict_success, validate_task_records, verify_actor_adapter, weight_sample,
)

ROOT = Path(__file__).resolve().parents[1]


def info(success=True, **extra):
    return dict(task_id=1, done=True, terminal_done=True, terminal_over=True,
                reward_type="gold_purchase" if success else "wrong_purchase", reward_valid=True,
                reward={}, policy_reward=.5, steps=4, assistant_tokens=20,
                actions=[{"tool": "buy"}], **extra)


@pytest.mark.parametrize("field,value", [("terminal_over", False), ("terminal_done", False),
    ("done", False), ("reward_valid", False), ("infrastructure_invalid", True),
    ("reward_unverifiable", True), ("reward_type", "valid_alternative_purchase")])
def test_strict_requires_complete_valid_gold(field, value):
    record = info()
    assert strict_success(record)
    record[field] = value
    assert not strict_success(record)


def test_record_counts_policy_tokens_not_observation(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_DIAGNOSTIC_ROLLOUT_DIR", str(tmp_path))
    monkeypatch.setenv("SHOPPING_PARENT_ADAPTER_SHA256", "abc")
    output = SimpleNamespace(response_mask=[1, 1, 0, 0, 0, 1], response_ids=[1]*6,
        reward_score=.6, prompt_ids=[2]*7, num_turns=4)
    annotated = annotate_rollout_statistics(info(), output)
    assert not list(tmp_path.glob("*.json"))
    class Batch:
        non_tensor_batch = {"shopping": [annotated]}
        def __len__(self):
            return 1
    persist_unpadded_validation_batch(Batch(), global_step=0)
    result = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert result["shopping"]["assistant_tokens"] == 3
    assert result["shopping"]["tool_and_template_tokens"] == 3
    assert result["shopping"]["policy_reward"] == .6
    assert result["adapter_sha256"] == "abc"


def test_fixed_denominator_and_adapter_identity():
    item = info()
    item["strict_success"] = True
    record = {"adapter_sha256": "abc", "shopping": item,
              "receipt_scope": "post_native_unpad"}
    assert len(validate_task_records([record], {1}, "abc")) == 1
    for records, tasks, sha in [([record], {1, 2}, "abc"), ([record, record], {1}, "abc"),
                                 ([record], {1}, "wrong"), ([record], {2}, "abc")]:
        with pytest.raises(ValueError):
            validate_task_records(records, tasks, sha)


def test_common_success_uses_identical_intersection():
    left = {1: info(), 2: info(False)}
    right = {1: info(), 2: info()}
    right[1]["steps"] = 2
    right[2]["steps"] = 100
    result = compare_tasks({"opd": left, "r0": right})
    assert result["common_success_tasks"] == [1]
    assert result["arms"]["r0"]["common_success_mean_steps"] == 2
    assert result["pairs"]["opd_to_r0"]["gains"] == [2]


def test_live_adapter_checks_all_values_with_dtype_conversion(tmp_path):
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)})
    module = Module()
    saved = {"lora_A.weight": module.lora_A.default.weight.detach().float()}
    # Production names contain a path separator before lora_A.
    outer = torch.nn.Module()
    outer.layer = module
    save_file({"layer." + k: v for k, v in saved.items()}, tmp_path / "adapter_model.safetensors")
    assert verify_actor_adapter(outer, tmp_path)["tensor_count"] == 1
    with torch.no_grad():
        outer.layer.lora_A.default.weight.add_(1)
    with pytest.raises(RuntimeError, match="differs"):
        verify_actor_adapter(outer, tmp_path)


def test_unfused_weight_samples_match_across_hf_vllm_prefixes():
    tensor = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    a = weight_sample("model.language_model.layers.0.mlp.down_proj.weight", tensor)
    b = weight_sample("language_model.model.layers.0.mlp.down_proj.weight", tensor.clone())
    assert samples_equal(a, b)
    b["values_bf16"][0] += 1
    assert not samples_equal(a, b)


def test_live_adapter_audit_unflattens_real_fsdp_on_cpu(tmp_path):
    if not hasattr(torch.distributed, "HashStore"):
        pytest.skip("this Torch build has no c10d HashStore; run the integration check on pinned Linux")
    from torch.distributed import HashStore, init_process_group, destroy_process_group
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    module = torch.nn.Module()
    module.layer = torch.nn.Module()
    module.layer.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False)})
    save_file({"layer.lora_A.weight": module.layer.lora_A.default.weight.detach().clone()},
              tmp_path / "adapter_model.safetensors")
    init_process_group("gloo", store=HashStore(), rank=0, world_size=1)
    try:
        wrapped = FSDP(module, device_id=torch.device("cpu"))
        assert any("_flat_param" in name for name, _ in wrapped.named_parameters())
        assert verify_actor_adapter(wrapped, tmp_path)["tensor_count"] == 1
    finally:
        destroy_process_group()


def test_eval_profile_is_greedy_read_only_and_cannot_accept_training_overrides(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("eval_entry", ROOT / "scripts/train_grpo.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    model, adapter = tmp_path / "model", tmp_path / "adapter"
    model.mkdir(); adapter.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"base")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "chat_template.jinja").write_text("template")
    contexts = tmp_path / "contexts.json"
    contexts.write_text("{}")
    monkeypatch.setattr("sys.argv", ["train", "--profile", "eval", "--model", str(model),
        "--adapter", str(adapter), "--identity-contexts", str(contexts), "--output", str(tmp_path / "out")])
    args = entry.parse_args()
    command, env = entry.build_command(args)
    for value in ["trainer.val_only=true", "trainer.save_freq=-1", "trainer.test_freq=-1",
                  "data.val_batch_size=1", "actor_rollout_ref.rollout.agent.num_workers=1",
                  "actor_rollout_ref.rollout.enable_prefix_caching=false",
                  "actor_rollout_ref.rollout.enforce_eager=true",
                  "actor_rollout_ref.rollout.val_kwargs.n=1", "shopping_evaluation.enable=true"]:
        assert value in command
    assert env["GRPO_VAL_FILE"].endswith("validation.parquet")
    assert env["SHOPPING_GIGPO_PROBE_DIR"] == ""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    search = (ROOT / "outputs/preparation/v1/runtime-wheel/source/verl/trainer/config").as_posix()
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(config_name="grpo", overrides=command[5:] + [f'hydra.searchpath=["file://{search}"]'])
    assert config.trainer.val_only and config.trainer.save_freq == -1
    assert config.data.val_batch_size == config.actor_rollout_ref.rollout.agent.num_workers == 1
    assert not config.actor_rollout_ref.rollout.enable_prefix_caching
    assert config.actor_rollout_ref.rollout.enforce_eager
    assert config.actor_rollout_ref.rollout.val_kwargs.n == 1
    assert not config.actor_rollout_ref.rollout.val_kwargs.do_sample
    args.hydra_overrides = ["--", "trainer.val_only=false"]
    with pytest.raises(SystemExit, match="eval accepts"):
        entry.build_command(args)


def test_identity_requests_repeat_fixed_training_context_without_mutating_task(tmp_path, monkeypatch):
    contexts = tmp_path / "contexts.json"
    contexts.write_text(json.dumps({"trigger_validation_task_id": 1,
        "contexts": [{"task_id": 53, "messages": [{"role": "user", "content": "training"}]}]}))
    monkeypatch.setenv("SHOPPING_DIAGNOSTIC_DIR", str(tmp_path / "receipts"))
    monkeypatch.setenv("SHOPPING_DIAGNOSTIC_CONTEXTS", str(contexts))
    calls = []
    async def template(messages, **kwargs):
        assert messages[0]["content"] == "training"
        return [3, 4]
    async def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(token_ids=[8], log_probs=[-.1],
            extra_fields={"prompt_logprobs": [[-.3], [0.]], "prompt_ids": [[4], [0]]})
    loop = SimpleNamespace(apply_chat_template=template, tool_schemas=[],
                           server_manager=SimpleNamespace(generate=generate))
    state = {"task_id": 1}
    data = SimpleNamespace(prompt_ids=[9], response_mask=[])
    asyncio.run(run_identity_audit(loop, state, data))
    asyncio.run(run_identity_audit(loop, state, data))
    # A different worker/session may receive a padded copy of the trigger task.
    asyncio.run(run_identity_audit(loop, {"task_id": 1}, data))
    assert len(calls) == 2 and data.prompt_ids == [9]
    assert calls[0]["prompt_ids"] == calls[1]["prompt_ids"] == [3, 4]
