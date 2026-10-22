"""Select predeclared probe/smoke rows from canonical GRPO parquet without retokenizing."""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = ROOT / "data/grpo"
    table = pq.read_table(folder / "train.parquet")
    rows = table.to_pylist()
    ids = [int(row["extra_info"]["task_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("canonical training task IDs must be unique")
    selection = json.loads((ROOT / "outputs/preparation/v1/task-selection.json").read_text(encoding="utf-8"))
    frozen = selection["pilot_ids"]
    if frozen != random.Random(selection["seed"]).sample(ids, len(frozen)):
        raise ValueError("subset selection differs from the predeclared random sample")
    evaluation_ids = {int(json.loads(line)["task_id"]) for line in
                      (ROOT / "data/evaluation/tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    if set(ids) & evaluation_ids:
        raise ValueError("training tasks overlap the canonical evaluation set")
    by_id = {task: index for index, task in enumerate(ids)}
    receipts = {}
    for name, count in (("smoke8", 8), ("probe32", 32), ("pilot64", 64)):
        subset = frozen[:count]
        output = folder / f"{name}.parquet"
        pq.write_table(table.take(pa.array([by_id[t] for t in subset])), output)
        (folder / f"{name}.jsonl").write_text(
            "".join(json.dumps({"task_id": task}) + "\n" for task in subset), encoding="utf-8")
        receipts[name] = {"task_ids": subset, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (ROOT / "outputs/preparation/v1/subset-manifest.json").write_text(
        json.dumps({"seed": selection["seed"], "selection": "uniform before rollouts",
                    "subsets": receipts}, indent=2), encoding="utf-8")
    print(json.dumps({name: len(item["task_ids"]) for name, item in receipts.items()}))


if __name__ == "__main__":
    main()
