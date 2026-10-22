import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import yaml

from shopping_grpo.training.grpo.adapter.runtime import current_runtime_state, make_runtime_state
from shopping_grpo.training.grpo.compat import _install_gigpo_actor_update, _install_trace_actor_update
from test_gigpo import batch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_entry", ROOT / "scripts/train_grpo.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class Config(dict):
    def __getattr__(self, name):
        return self[name]


def config(enabled):
    return Config(shopping_gigpo={"enable": enabled, "omega": 1., "gamma": 1., "reward_tolerance": 1e-8},
                  shopping_trace={"enable": False}, algorithm={"adv_estimator": "grpo",
                  "norm_adv_by_std_in_grpo": False, "use_kl_in_reward": False},
                  actor_rollout_ref=Config(rollout=Config(n=4)))


@pytest.fixture
def parent(tmp_path):
    model, adapter = tmp_path / "base", tmp_path / "adapter"
    model.mkdir()
    adapter.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"test base")
    (adapter / "adapter_model.safetensors").write_bytes(b"test adapter")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32}))
    (adapter / "chat_template.jinja").write_text("template")
    return model, adapter


def args(monkeypatch, parent, tmp_path, **options):
    model, adapter = parent
    argv = ["train_grpo.py", "--model", str(model), "--adapter", str(adapter),
            "--output", str(tmp_path / "output")]
    for name, value in options.items():
        argv.extend(["--" + name.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", argv)
    return entry.parse_args()


def test_main_arms_share_parent_budget_and_differ_only_in_method(monkeypatch, parent, tmp_path):
    first = args(monkeypatch, parent, tmp_path, profile="main", method="grpo")
    second = args(monkeypatch, parent, tmp_path, profile="main", method="gigpo")
    left, le = entry.build_command(first)
    right, re = entry.build_command(second)
    assert [(a, b) for a, b in zip(left, right) if a != b] == [
        ("shopping_gigpo.enable=false", "shopping_gigpo.enable=true")]
    assert le == re
    assert "trainer.total_training_steps=64" in left
    assert "trainer.resume_mode=disable" in left
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed=42" in left
    assert le["GRPO_TRAIN_FILE"].endswith("train.parquet")
    assert le["SHOPPING_GIGPO_PROBE_DIR"] == ""


def test_probe_uses_training_subset_and_validation_only(monkeypatch, parent, tmp_path):
    command, env = entry.build_command(args(monkeypatch, parent, tmp_path, profile="probe"))
    assert "trainer.val_only=true" in command
    assert "trainer.val_before_train=true" in command
    assert "actor_rollout_ref.rollout.val_kwargs.n=4" in command
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed=42" in command
    assert env["GRPO_VAL_FILE"].endswith("probe32.parquet")
    assert env["SHOPPING_GIGPO_PROBE_DIR"].endswith("probe_rollouts")


def test_dry_run_creates_no_output(monkeypatch, parent, tmp_path, capsys):
    options = args(monkeypatch, parent, tmp_path, profile="main")
    options.dry_run = True
    monkeypatch.setattr(entry, "parse_args", lambda: options)
    entry.main()
    assert not (tmp_path / "output").exists()
    assert json.loads(capsys.readouterr().out)["profile"] == "main"


@pytest.mark.parametrize("preflight_status", [0, 9])
def test_launch_checks_exact_overrides_and_stops_on_preflight_failure(
    monkeypatch, parent, tmp_path, preflight_status
):
    options = args(monkeypatch, parent, tmp_path, profile="main", method="gigpo")
    options.hydra_overrides = ["--", "actor_rollout_ref.actor.optim.lr=1e-6"]
    command, environment = entry.build_command(options)
    monkeypatch.setattr(entry, "parse_args", lambda: options)
    calls = []

    def call(argv, *, cwd, env):
        calls.append(argv)
        assert cwd == ROOT and env == environment
        return preflight_status if len(calls) == 1 else 0

    monkeypatch.setattr(entry.subprocess, "call", call)
    with pytest.raises(SystemExit) as stopped:
        entry.main()
    assert stopped.value.code == preflight_status
    assert calls[0] == [sys.executable, str(ROOT / "scripts/check_grpo_runtime.py"), *command[5:]]
    assert len(calls) == (2 if preflight_status == 0 else 1)
    if preflight_status == 0:
        assert calls[1] == command
    audit = json.loads((tmp_path / "output/launch.json").read_text(encoding="utf-8"))
    assert audit["command"] == command


def test_missing_template_is_rejected(monkeypatch, parent, tmp_path):
    (parent[1] / "chat_template.jinja").unlink()
    with pytest.raises(SystemExit, match="chat_template"):
        entry.build_command(args(monkeypatch, parent, tmp_path))


@pytest.mark.parametrize("driver_template,processor_template,error", [
    (None, None, "driver tokenizer"),
    ("sft", None, "processor"),
    ("different", "sft", "driver tokenizer"),
    ("sft", "different", "processor"),
    ("sft", "sft", None),
])
def test_native_driver_and_processor_keep_parent_template(driver_template, processor_template, error):
    spec = importlib.util.spec_from_file_location("prompt_checker", ROOT / "scripts/check_grpo_runtime.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    driver = SimpleNamespace(chat_template=driver_template)
    processor = SimpleNamespace(chat_template=processor_template)
    actor = SimpleNamespace(chat_template="sft")
    if error:
        with pytest.raises(SystemExit, match=error):
            checker.validate_runtime_chat_templates(driver, processor, actor)
    else:
        checker.validate_runtime_chat_templates(driver, processor, actor)
        checker.validate_runtime_chat_templates(driver, None, actor)


def test_saved_subset_ids_and_final_data_are_disjoint():
    def load(name):
        return {json.loads(line)["task_id"] for line in (ROOT / name).read_text().splitlines() if line}
    train = load("data/grpo/train.jsonl")
    final = load("data/evaluation/tasks.jsonl")
    probe = load("data/grpo/probe32.jsonl")
    assert len(probe) == 32 and probe <= train and not train & final


def test_hook_disabled_is_noop_and_install_is_idempotent(monkeypatch):
    module = ModuleType("verl.trainer.ppo.ray_trainer")
    calls = []

    class Trainer:
        def _update_actor(self, data):
            calls.append(data.batch["advantages"].clone())
            return SimpleNamespace(meta_info={"metrics": {"actor/loss": 0.}})

    module.RayPPOTrainer = Trainer
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.ray_trainer", module)
    for _ in range(3):
        _install_trace_actor_update()
        _install_gigpo_actor_update()
    trainer = Trainer()
    data = batch()
    before = data.batch["advantages"].clone()
    trainer.config = config(False)
    data.non_tensor_batch["trace_target"] = ["private"] * 4
    trainer._update_actor(data)
    assert "trace_target" not in data.non_tensor_batch
    assert torch.equal(before, data.batch["advantages"])
    trainer.config = config(True)
    output = trainer._update_actor(data)
    assert torch.equal(data.batch["advantages"], before * 2)
    assert len(calls) == 2
    assert "gigpo/effective_noninitial_turns" in output.meta_info["metrics"]


@pytest.mark.parametrize("sampling,expected_max", [
    ({"temperature": .7, "top_p": .9}, 512),
    ({"max_tokens": 100, "stop_token_ids": [99]}, 100),
    ({"max_tokens": 1000}, 512),
])
def test_generation_boundary_records_real_response_mask_offsets(monkeypatch, sampling, expected_max):
    native = ModuleType("verl.experimental.agent_loop.tool_agent_loop")
    native.AgentState = SimpleNamespace(TERMINATED="end")
    calls = []

    class ToolAgentLoop:
        async def _handle_generating_state(self, data, params, **kwargs):
            calls.append(params)
            data.response_ids = [9, 8]
            data.response_mask.extend([1, 1])
            data.prompt_ids.extend([9, 8])
            return "tools"

    native.ToolAgentLoop = ToolAgentLoop
    monkeypatch.setitem(sys.modules, "verl.experimental.agent_loop.tool_agent_loop", native)
    name = "shopping_grpo.training.grpo.adapter.agent_loop"
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/shopping_grpo/training/grpo/adapter/agent_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loop = object.__new__(module.ShoppingToolAgentLoop)
    loop.gigpo_record_turns = True
    loop.context_compaction_enable = False
    loop.context_window_tokens = 24576
    loop.context_generation_reserve_tokens = 512
    loop.context_safety_margin_tokens = 512
    loop.tokenizer = SimpleNamespace(eos_token_id=248044, unk_token_id=None,
        convert_tokens_to_ids=lambda marker: 248046 if marker == "<|endoftext|>" else 248044)
    state = make_runtime_state(1, 35)
    state["latest_observation"] = "public page"
    token = current_runtime_state.set(state)
    try:
        data = SimpleNamespace(prompt_ids=[1, 2, 3, 4], response_mask=[1, 1, 0, 0],
                               response_logprobs=[], response_ids=[])
        result = asyncio.run(loop._handle_generating_state(data, sampling))
    finally:
        current_runtime_state.reset(token)
    assert result == "tools"
    assert state["gigpo_turns"][0]["start"] == 4
    assert state["gigpo_turns"][0]["end"] == 6
    assert len(data.response_mask) == 6
    assert calls[0]["max_tokens"] == expected_max
    assert set(calls[0]["stop_token_ids"]) == {248044, 248046, *sampling.get("stop_token_ids", [])}
    for key in ("temperature", "top_p"):
        assert calls[0].get(key) == sampling.get(key)


def test_configs_keep_reward_and_sampling_fixed():
    config = yaml.safe_load((ROOT / "configs/grpo.yaml").read_text())
    agent = yaml.safe_load((ROOT / "configs/agent_loop.yaml").read_text())[0]
    assert config["shopping_gigpo"]["gamma"] == 1.
    assert not config["algorithm"]["norm_adv_by_std_in_grpo"]
    assert config["actor_rollout_ref"]["rollout"]["n"] == 4
    assert agent["reward_mode"] == "constraint_aware"
    assert agent["context_compaction_enable"] is False
