"""Rescore the frozen training probe on CPU; never run model generation."""
import argparse
import hashlib
import json
from pathlib import Path

from shopping_grpo.training.grpo.reward_audit import audit_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.rollouts.glob("*.json"))]
    report = audit_probe(records, expected_tasks=set(plan["probe_task_ids"]),
                         parent_sha=plan["parent_adapter_sha256"])
    report["plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit("preserve the previous audit; choose a new output file")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ready_for_smoke": report["ready_for_smoke"],
                      "changed_rollouts": report["changed_rollouts"],
                      "blocking_reasons": report["blocking_reasons"]}, ensure_ascii=False))
    if not report["ready_for_smoke"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
