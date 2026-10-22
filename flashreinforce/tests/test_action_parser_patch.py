"""Exercise the native action parser with real tool-serialized option strings."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import re

import pytest

from shopping_grpo.environment.tools import tool_call_to_action

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT.parent / "sft/environments/ShopSimulator/shop_env"


def native_parser(text):
    node = next(node for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name == "parse_action")
    namespace = {"re": re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "native/engine.py", "exec"), namespace)
    return namespace["parse_action"]


@pytest.fixture
def parser():
    spec = importlib.util.spec_from_file_location("parser_patch", ROOT / "scripts/patch_shop_action_parser.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = module.patched_source((NATIVE / "web_agent_site/engine/engine.py").read_bytes()).decode("utf-8")
    return native_parser(text)


@pytest.mark.parametrize("value", ["镜框+1.67防蓝光镜片[0-800度]", "套装[主机[黑色]+配件]", "纯棉 XL"])
def test_native_option_roundtrip_preserves_outer_action_and_complete_value(parser, value):
    action = tool_call_to_action("select_option", {"value": value})
    assert parser(action) == ("click", value)


def test_native_search_keeps_literal_brackets(parser):
    query = "镜片[0-800度] 防蓝光"
    assert parser(tool_call_to_action("search_products", {"query": query})) == ("search", query)


@pytest.mark.parametrize("action,expected", [("click[Buy Now]", ("click", "Buy Now")),
    ("finish", ("finish", None)), ("search[棉线 黑色]", ("search", "棉线 黑色"))])
def test_existing_valid_actions_keep_their_meaning(parser, action, expected):
    assert parser(action) == expected


@pytest.mark.parametrize("action", ["click[a] trailing", "click[]", "click[unterminated"])
def test_malformed_actions_do_not_turn_into_partial_clicks(parser, action):
    assert parser(action) == (action, None)


def test_parser_repair_uses_only_existing_option_price(parser, monkeypatch):
    monkeypatch.syspath_prepend(str(NATIVE))
    from web_agent_site.engine.variant_price import resolve_variant_price
    value = "镜框+1.67防蓝光镜片[0-800度]"
    product = {"customization_options": {"镜片折射率": [
        {"value": value, "price": 716}, {"value": "单镜框", "price": 316}]}}
    name, selected = parser(tool_call_to_action("select_option", {"value": value}))
    options = {"镜片折射率": selected} if name == "click" else {}
    assert resolve_variant_price(product, options)["price"] == 716


def test_runtime_patch_has_explicit_identity_and_cannot_hide_engine_drift(tmp_path):
    spec = importlib.util.spec_from_file_location("checker", ROOT / "scripts/check_grpo_runtime.py")
    checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
    hashes = {}
    for name, relative in checker.CURRENT_RUNTIME_FILES.items():
        p = tmp_path / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"unchanged")
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("parser_patch", ROOT / "scripts/patch_shop_action_parser.py")
    patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
    original = (NATIVE / "web_agent_site/engine/engine.py").read_bytes()
    repaired = patch.patched_source(original)
    engine = tmp_path / "environments/ShopSimulator/shop_env/web_agent_site/engine/engine.py"
    engine.parent.mkdir(parents=True, exist_ok=True); engine.write_bytes(repaired)
    hashes["engine.py"] = hashlib.sha256(repaired).hexdigest()
    manifest = {"lease_contract": "explicit-client-release-v1", "runtime_files_sha256": hashes,
                "runtime_patches": {"action_parser": {"version": patch.VERSION,
                    "source_sha256": patch.ORIGINAL_SHA256, "patched_sha256": hashes["engine.py"]}}}
    checker.validate_reward_runtime_files(manifest, tmp_path)
    bad = deepcopy(manifest); bad["runtime_patches"]["action_parser"]["patched_sha256"] = "0"*64
    with pytest.raises(SystemExit, match="unrecognized"):
        checker.validate_reward_runtime_files(bad, tmp_path)
    engine.write_bytes(b"drift")
    with pytest.raises(SystemExit, match="hash mismatch"):
        checker.validate_reward_runtime_files(manifest, tmp_path)
    assert hashlib.sha256(original).hexdigest() == patch.ORIGINAL_SHA256
