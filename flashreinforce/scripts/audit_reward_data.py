"""CPU audit of training-only explicit option coverage and counterfactual variants."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys

from prepare_reward_experiment import ROOT, CANDIDATE, ids, sha
from shopping_grpo.training.grpo.adapter.runtime import validate_reward
from shopping_grpo.training.grpo.reward_overlay import VERSION, apply_overlay


def native_function(path, name, namespace):
    module = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace[name]


def audit(source, products_path):
    sys.path.insert(0, str(source))
    from web_agent_site.engine.constraints import explicit_budget_from_instruction
    from web_agent_site.engine.reward_features import compile_reward_features
    from web_agent_site.engine.reward import evaluate_purchase
    from web_agent_site.engine.variant_price import candidate_options_for_evaluation, resolve_variant_price
    with gzip.open(products_path, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    engine = source / "web_agent_site/engine/engine.py"
    goal_source = source / "web_agent_site/engine/goal.py"
    clean = native_function(engine, "clean_product_keys", {"print": lambda *_: None})
    clean(products)
    # Match the pinned loader's ASIN filtering and fields used by goal assembly.
    unique, by_asin, normalized = set(), {}, []
    for product in products:
        asin = product["asin"]
        if asin == "nan" or len(asin) > 20 or asin in unique:
            continue
        unique.add(asin)
        product.setdefault("query", "")
        product.setdefault("reason_key", None)
        product.setdefault("user_persona", None)
        product["pricing"] = (product.get("pricing") or [100.])[:2]
        normalized.append(product)
        by_asin[asin] = product
    get_goals = native_function(goal_source, "get_existed_goals", {
        "defaultdict": defaultdict, "compile_reward_features": compile_reward_features,
        "explicit_budget_from_instruction": explicit_budget_from_instruction,
        "print": lambda *_: None})
    goals = get_goals(normalized, {}, if_persona=False)
    original_ids = ids(ROOT / "data/grpo/train.jsonl")
    candidate_ids = set(ids(ROOT / CANDIDATE / "train953.jsonl"))
    resolved, prepared = set(), {}
    for task in original_ids:
        goal = goals[task]
        product = by_asin[goal["asin"]]
        selected, resolution = candidate_options_for_evaluation(product, goal["required_options_by_key"])
        price = resolve_variant_price(product, selected)
        if not goal["unresolved_option_requirements"] and resolution["status"] == "pass" and price["status"] == "pass":
            resolved.add(task)
        if task in candidate_ids:
            prepared[task] = (product, goal, selected)
    if resolved != candidate_ids:
        raise ValueError("native goal/price reconstruction differs from the frozen 953 pool: "
                         + str({"extra": sorted(resolved-candidate_ids), "missing": sorted(candidate_ids-resolved)}))
    statuses, active, reference_native, synthetic_applied, lower_than_stop = Counter(), [], Counter(), [], []
    samples = []
    for task in sorted(candidate_ids):
        product, goal, selected = prepared[task]
        result = evaluate_purchase(product, goal, selected_options=selected)
        detail = validate_reward(result.to_dict(), instruction=goal["instruction_text"])
        evidence = detail["necessary_options"]
        statuses[evidence["status"]] += 1
        reference_native[result.reward_type] += 1
        if evidence["active"]:
            active.append(task)
            if len(samples) < 20:
                samples.append({"task_id": task, "instruction": goal["instruction_text"],
                                "necessary_options": evidence})
        for axis, entries in (product.get("customization_options") or {}).items():
            if axis not in selected or not isinstance(entries, list):
                continue
            wrong = next((entry.get("value") for entry in entries if isinstance(entry, dict)
                          and entry.get("value") and entry["value"] != selected[axis]), None)
            if wrong is None:
                continue
            changed = deepcopy(selected)
            changed[axis] = wrong
            counterfactual = evaluate_purchase(product, goal, selected_options=changed)
            details = validate_reward(counterfactual.to_dict(), instruction=goal["instruction_text"])
            reward = {"total": counterfactual.reward, "sampling_invalid": not counterfactual.reward_valid}
            overlay = apply_overlay(reward, details, mode=VERSION, is_validation=False)
            if overlay["applied"]:
                synthetic_applied.append(task)
                if overlay["training_reward"] < -.15:
                    lower_than_stop.append(task)
    if not synthetic_applied or len(lower_than_stop) != len(synthetic_applied):
        raise ValueError("no applicable counterfactuals or necessary-option ordering failed")
    return {"schema_version": 1, "overlay_version": VERSION, "training_tasks": len(candidate_ids),
            "goal_mapping_verified_by_exact_price953_selection": True,
            "product_data_sha256": sha(products_path), "native_goal_source_sha256": sha(goal_source),
            "reference_status_counts": dict(statuses), "reference_native_outcome_counts": dict(reference_native),
            "tasks_with_explicit_supported_options": len(active), "active_task_ids": active,
            "synthetic_changed_variants": len(synthetic_applied),
            "synthetic_changed_task_count": len(set(synthetic_applied)),
            "all_changed_variants_below_eligible_stop": True,
            "manual_review_samples": samples, "model_generation_calls": 0,
            "evaluation_tasks_used_for_coefficient_selection": False,
            "scope": "native training-task reference/counterfactual audit; not on-policy coverage or training effectiveness"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-source", type=Path, required=True, help="unchanged ShopSimulator shop_env directory")
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("preserve the previous audit; choose a new output file")
    report = audit(args.env_source.resolve(), args.products.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("training_tasks", "tasks_with_explicit_supported_options",
        "synthetic_changed_variants", "synthetic_changed_task_count", "reference_status_counts")}))


if __name__ == "__main__":
    main()
