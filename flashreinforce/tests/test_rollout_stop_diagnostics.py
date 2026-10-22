"""Run the pinned native stop branches through the real Shopping adapter."""
import ast
import asyncio
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment, current_runtime_state, make_runtime_state, reward_breakdown,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def loop(monkeypatch):
    source = ROOT / "outputs/preparation/v1/runtime-wheel/source/verl/experimental/agent_loop/tool_agent_loop.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    state = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "AgentState")
    base = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "ToolAgentLoop")
    base.bases = []
    base.decorator_list = []
    base.body = [n for n in base.body if isinstance(n, ast.AsyncFunctionDef)
                 and n.name in ("_handle_generating_state", "_handle_processing_tools_state")]
    for function in base.body:
        function.decorator_list = []
    from enum import Enum
    namespace = dict(Enum=Enum, asyncio=asyncio, simple_timer=lambda *a, **kw: nullcontext(),
                     SPEC_DECODE_EXTRA_KEYS=())
    header = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[header, state, base], type_ignores=[])),
                 str(source), "exec"), namespace)
    native = ModuleType("verl.experimental.agent_loop.tool_agent_loop")
    native.AgentState, native.ToolAgentLoop = namespace["AgentState"], namespace["ToolAgentLoop"]

    async def call_tool(self, *args):
        return SimpleNamespace(text="public observation", image=None, video=None), 0., {}
    native.ToolAgentLoop._call_tool = call_tool
    monkeypatch.setitem(sys.modules, native.__name__, native)
    spec = importlib.util.spec_from_file_location("tested_shopping_loop", ROOT / "src/shopping_grpo/training/grpo/adapter/agent_loop.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    result = object.__new__(adapter.ShoppingToolAgentLoop)
    result.gigpo_record_turns = False
    result.context_compaction_enable = False
    result.context_window_tokens = 24576
    result.context_generation_reserve_tokens = 512
    result.context_safety_margin_tokens = 512
    result.response_length = 8
    result.max_assistant_turns = 40
    result.max_user_turns = 40
    result.max_parallel_calls = 1
    result.tools = {}
    result.tool_parser_name = "qwen3_coder"
    result.tokenizer = SimpleNamespace(eos_token_id=None, unk_token_id=None,
        convert_tokens_to_ids=lambda _: None, decode=lambda ids, **kw: "last public assistant output")

    async def template(*a, **kw):
        return [7, 7, 7]
    async def generate(**kw):
        return SimpleNamespace(token_ids=[9, 8], log_probs=[-.1, -.2], num_preempted=0,
                               extra_fields={}, routed_experts=None)
    async def parse(*a):
        return None, []
    result.apply_chat_template = template
    result.server_manager = SimpleNamespace(generate=generate)
    result.tool_parser = SimpleNamespace(stop_token_ids=[], extract_tool_calls=parse)
    return result


def data():
    return SimpleNamespace(prompt_ids=[1]*6, response_ids=[], response_mask=[1, 0, 0],
        response_logprobs=[-.1, 0., 0.], metrics={}, request_id="cpu-only", image_data=None,
        video_data=None, audio_data=None, mm_processor_kwargs={}, assistant_turns=0, user_turns=0,
        extra_fields={}, tool_calls=[], tool_rewards=[], messages=[], tools_kwargs={})


@pytest.mark.parametrize("setting,reason", [
    ("response", "response_token_limit_after_generation"),
    ("assistant", "assistant_turn_limit"), ("user", "user_turn_limit"),
    ("no_tool", "assistant_no_tool_call"),
])
def test_actual_native_generation_stop_is_identified_and_stays_invalid(loop, setting, reason):
    item = data()
    if setting == "response": loop.response_length = 5
    if setting == "assistant": loop.max_assistant_turns = 1
    if setting == "user": loop.max_user_turns = 1; item.user_turns = 1
    state = make_runtime_state(18330, 35)
    token = current_runtime_state.set(state)
    try:
        asyncio.run(loop._handle_generating_state(item, {}))
    finally:
        current_runtime_state.reset(token)
    assert state["agent_loop_stop"]["reason"] == reason
    assert state["agent_loop_stop"]["last_assistant_token_ids"] == [9, 8]
    assert state["agent_loop_stop"]["assistant_tokens"] == 3
    assert item.response_logprobs == [-.1, 0., 0., -.1, -.2]
    assert reward_breakdown(state)["sampling_invalid"] is True
    assert reward_breakdown(state)["total"] == 0


def test_actual_native_tool_observation_budget_stop(loop):
    item = data()
    item.response_mask = [1, 0, 0, 1, 1]
    item.response_ids = [9, 8]
    item.tool_calls = [SimpleNamespace(name="search_products", arguments="{}")]
    state = make_runtime_state(18330, 35)
    token = current_runtime_state.set(state)
    try:
        asyncio.run(loop._handle_processing_tools_state(item))
    finally:
        current_runtime_state.reset(token)
    assert state["agent_loop_stop"]["reason"] == "response_token_limit_before_tool_observation"
    assert item.response_mask == [1, 0, 0, 1, 1]  # Never fabricate observation tokens/logps.
    assert not state["done"] and reward_breakdown(state)["sampling_invalid"]


def test_guard_feedback_offers_only_current_targets_and_records_rejected_call():
    from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
    state = make_runtime_state(192, 35)
    state["latest_observation"] = '1|111111111111|current item\n\n可点击的按钮: ["Back to Search"]'
    tool = ShopSimulatorTool({}, {"function": {"name": "open_product", "parameters": {}}})
    a, b = current_runtime_state.set(state), current_environment.set(object())
    try:
        response, _, _ = asyncio.run(tool.execute("id", {"asin": "999999999999"}))
    finally:
        current_runtime_state.reset(a)
        current_environment.reset(b)
    assert "111111111111" in response.text and "back_to_search" in response.text
    assert "Back to Search" in response.text
    assert state["guard_rejection_details"][0]["parameters"] == {"asin": "999999999999"}
    assert state["guard_rejection_details"][0]["observation"] == state["latest_observation"]
    assert not state["steps"] and state["guard_rejection_count"] == 1
