"""Exercise the pinned native padding call site and the real shopping AgentLoop."""
import ast
import asyncio
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from shopping_grpo.training.grpo.adapter.runtime import make_runtime_state

ROOT = Path(__file__).resolve().parents[1]


def patched_source():
    original = ROOT / "outputs/preparation/v1/runtime-wheel/source/verl/trainer/ppo/ray_trainer.py"
    source = original.read_text(encoding="utf-8").splitlines(keepends=True)
    patch = (ROOT / "patches/verl-0.8.0-shopping-dynamic-sampling.patch").read_text(encoding="utf-8").splitlines(keepends=True)
    result, cursor, index = [], 0, 0
    while index < len(patch):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", patch[index])
        if not match:
            index += 1
            continue
        count = int(match[2] or 1)
        start = int(match[1]) - (1 if count else 0)
        result.extend(source[cursor:start])
        cursor, index = start, index + 1
        while index < len(patch) and not patch[index].startswith("@@"):
            line = patch[index]
            if line.startswith((" ", "-")):
                assert source[cursor].rstrip("\r\n") == line[1:].rstrip("\r\n")
                cursor += 1
            if line.startswith((" ", "+")):
                result.append(line[1:])
            index += 1
    return "".join(result + source[cursor:])


class Proto:
    def __init__(self, infos):
        self.non_tensor_batch = {"shopping": np.array(infos, dtype=object)}

    def __len__(self):
        return len(self.non_tensor_batch["shopping"])

    def __getitem__(self, selection):
        return Proto(self.non_tensor_batch["shopping"][selection].tolist())

    @staticmethod
    def concat(items):
        return Proto([info for item in items for info in item.non_tensor_batch["shopping"]])


def native_validation_batch():
    # Execute the actual pinned pad/generate/unpad statements, including any
    # project receipt hook, without importing Ray/CUDA into the CPU test.
    tree = ast.parse(patched_source())
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    validate = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "_validate")
    loop = next(n for n in validate.body if isinstance(n, ast.For))
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and "pad_dataproto_to_divisor" in ast.unparse(n))
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Expr)
               and "validation generation end" in ast.unparse(n))
    function = ast.parse("def validate_batch(self, test_gen_batch):\n    pass").body[0]
    function.body = loop.body[start:end] + [ast.parse("return test_output_gen_batch").body[0]]
    protocol = ast.parse((ROOT / "outputs/preparation/v1/runtime-wheel/source/verl/protocol.py").read_text(encoding="utf-8"))
    functions = [n for n in protocol.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"pad_dataproto_to_divisor", "unpad_dataproto"}]
    namespace = {"DataProto": Proto, "size_divisor": 8}
    # The divisor assignment sits just before the extracted native statements.
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions + [function], type_ignores=[])),
                 "pinned_native_validation", "exec"), namespace)
    return namespace["validate_batch"]


def test_native_validation_saves_two_real_rows_not_eight_padding_copies(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING_DIAGNOSTIC_ROLLOUT_DIR", str(tmp_path))
    monkeypatch.setenv("SHOPPING_PARENT_ADAPTER_SHA256", "test-parent")
    native = ModuleType("verl.experimental.agent_loop.tool_agent_loop")
    native.AgentState = SimpleNamespace(TERMINATED="end")

    class ToolAgentLoop:
        async def run(self, *_args, **_kwargs):
            return SimpleNamespace(response_mask=[1, 0, 1], response_ids=[1, 2, 3],
                reward_score=0., prompt_ids=[4], num_turns=2, extra_fields={})

    native.ToolAgentLoop = ToolAgentLoop
    monkeypatch.setitem(sys.modules, native.__name__, native)
    spec = importlib.util.spec_from_file_location("shopping_test_loop", ROOT /
        "src/shopping_grpo/training/grpo/adapter/agent_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Session:
        trace_target = None

        def __init__(self, **kwargs):
            pass

        async def start(self, task):
            state = make_runtime_state(task, 35)
            state.update(done=True, terminal_result={"done": True, "over": True},
                reward_version="shopsimulator-reward-v3", reward_type="gold_purchase", final_reward=1.)
            return state

        async def close(self):
            pass

    monkeypatch.setattr(module, "ShopSimulatorSession", Session)
    agent = object.__new__(module.ShoppingToolAgentLoop)
    agent.base_url, agent.timeout, agent.max_steps = "unused", 1, 35
    agent.required_environment_version, agent.env_factory = None, None
    agent.reward_mode, agent.gigpo_record_turns = "constraint_aware", False

    class Manager:
        def generate_sequences(self, padded):
            infos = []
            for item in padded.non_tensor_batch["shopping"]:
                output = asyncio.run(agent.run({}, extra_info={"task_id": item["task_id"]}))
                infos.append(output.extra_fields["shopping"])
            return Proto(infos)

    trainer = SimpleNamespace(use_rm=False, async_rollout_manager=Manager(), global_steps=0)
    result = native_validation_batch()(trainer, Proto([{"task_id": 1}, {"task_id": 2}]))
    assert len(result) == 2
    receipts = [json.loads(p.read_text(encoding="utf-8")) for p in tmp_path.glob("task-*.json")]
    assert len(receipts) == 2, "native padding copies must not enter the fixed evaluation denominator"
    assert [r["shopping"]["task_id"] for r in receipts].count(1) == 1
    assert all(r["receipt_scope"] == "post_native_unpad" for r in receipts)
