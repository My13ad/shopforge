"""Audit no-gradient probe receipts without loading a model or contacting APIs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from shopping_grpo.training.grpo.probe import diagnose_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="Frozen probe32.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    launch = json.loads((args.run / "launch.json").read_text(encoding="utf-8"))
    if launch.get("profile") != "probe":
        raise SystemExit("only no-gradient probe runs are accepted")
    tasks = {int(json.loads(line)["task_id"]) for line in args.tasks.read_text(
        encoding="utf-8").splitlines() if line.strip()}
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((args.run / "probe_rollouts").glob("*.json"))]
    report = diagnose_probe(records, expected_tasks=tasks,
                            parent_sha=launch["parent_adapter_sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
