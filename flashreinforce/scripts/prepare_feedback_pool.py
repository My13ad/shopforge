#!/usr/bin/env python3
"""Mine repeated native failures from training receipts; export only sampling IDs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from shopping_grpo.training.grpo.diagnostics import strict_success
from shopping_grpo.training.grpo.dynamic_sampling import extract_shopping_group_signals
from shopping_grpo.training.grpo.feedback_sampling import PROTOCOL, load_manifest, sha256


def mine(train_ids, receipt_paths, *, size=192):
    stats = {task: Counter() for task in train_ids}
    seen = set()
    for path in receipt_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") != "flash_batch_admission":
                continue
            rows = event["rollouts"]
            invalid = extract_shopping_group_signals(rows)[2]
            for row, bad in zip(rows, invalid, strict=True):
                task, uid = row["task_id"], row["uid"]
                if task not in stats:
                    raise ValueError("training receipt includes a task outside the approved train pool")
                if uid in seen:
                    raise ValueError("duplicate training rollout receipt")
                seen.add(uid)
                count = stats[task]
                count["seen"] += 1
                if bad or row.get("reward_valid") is not True or row.get("reward_detail", {}).get("reward_valid") is not True:
                    count["invalid_diagnostic_only"] += 1
                    continue
                count["valid"] += 1
                if strict_success(row):
                    count["success"] += 1
                    continue
                count["failure"] += 1
                reason, detail = row.get("termination_reason"), row.get("reward_detail", {})
                if reason == "repeat_loop":
                    count["ineffective_exploration"] += 1
                if reason == "partial_alternative_purchase":
                    count["option_constraints" if detail.get("target_asin_match") else "candidate_comparison"] += 1
                if reason == "wrong_purchase":
                    count["budget_or_category"] += 1
                if row.get("guard_rejections", 0):
                    count["guard_recovery"] += 1
    repeated_failures = [t for t, c in stats.items() if c["valid"] >= 4 and c["failure"] >= 2 and c["failure"] / c["valid"] >= .35]
    candidates = [t for t in repeated_failures if stats[t]["success"] >= 1]
    # Beta(1,1) shrinkage reduces the influence of sparse observations. This is
    # a sampling heuristic, not an estimate of a particular checkpoint's accuracy.
    score = lambda t: ((stats[t]["failure"] + 1) / (stats[t]["valid"] + 2), stats[t]["failure"], -t)
    candidates.sort(key=score, reverse=True)
    if len(candidates) < size:
        raise ValueError("not enough repeatedly difficult training tasks")
    hard = sorted(candidates[:size])
    return hard, {"eligible_tasks": len(candidates), "persistent_failure_task_ids": sorted(set(repeated_failures) - set(candidates)),
                  "rules": {"minimum_valid": 4, "minimum_failures": 2, "minimum_strict_successes": 1, "minimum_failure_fraction": .35, "ranking": "Beta(1,1) posterior mean failure; failures descending; ID ascending"},
                  "tasks": [{"task_id": t, "hard_member": t in hard, "difficulty_score": score(t)[0], **dict(stats[t])} for t in sorted(stats)]}


def main():
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=192)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    train = [int(r["extra_info"]["task_id"]) for r in pq.read_table(args.train_data, columns=["extra_info"]).to_pylist()]
    hard, report = mine(train, args.receipts, size=args.size)
    path = args.output / "sampling-manifest.json"
    path.write_text(json.dumps({"protocol": PROTOCOL, "train_sha256": sha256(args.train_data), "train_task_ids": train, "hard_task_ids": hard}, indent=2) + "\n", encoding="utf-8")
    load_manifest(path, args.train_data)
    report.update(sources=[{"file": str(p), "sha256": sha256(p)} for p in args.receipts],
                  train_sha256=sha256(args.train_data), manifest_sha256=sha256(path),
                  scope="training receipts only; no Final200 traces, prompts or answers read; invalid rewards excluded from mining evidence")
    (args.output / "mining-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train), "hard": len(hard), "eligible": report["eligible_tasks"], "manifest_sha256": sha256(path)}))


if __name__ == "__main__":
    main()
