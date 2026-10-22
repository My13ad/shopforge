"""Freeze the reward experiment inputs on CPU; never load or train a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shopping_grpo.training.grpo.reward_overlay import VERSION

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "6d696b665b12d9d2941efe0e71376678729ccc245efea7d26809c32ae80629ef"
CANDIDATE = "outputs/preparation/v1/candidate-price953"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ids(path):
    rows = [int(json.loads(line)["task_id"]) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError("duplicate task IDs: " + str(path))
    return rows


def prepare(root=ROOT):
    candidate = root / CANDIDATE
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    original = set(ids(root / "data/grpo/train.jsonl"))
    validation = set(ids(root / "data/grpo/validation.jsonl"))
    final = set(ids(root / "data/evaluation/tasks.jsonl"))
    hashes, pools = {}, {}
    for name, count in (("train953", 953), ("probe60", 60), ("smoke8", 8)):
        pool = ids(candidate / f"{name}.jsonl")
        declared = manifest["subsets"][name]
        if len(pool) != count or pool != declared["task_ids"] or declared["count"] != count:
            raise ValueError("candidate subset differs from frozen selection: " + name)
        parquet = candidate / f"{name}.parquet"
        if sha(parquet) != declared["sha256"]:
            raise ValueError("candidate parquet hash mismatch: " + name)
        for path in (parquet, candidate / f"{name}.jsonl"):
            hashes[path.relative_to(root).as_posix()] = sha(path)
        pools[name] = set(pool)
    if len(original) != 1000 or len(validation) != 50 or len(final) != 200:
        raise ValueError("unexpected canonical data counts")
    if not pools["train953"] <= original or not (pools["probe60"] | pools["smoke8"]) <= pools["train953"]:
        raise ValueError("candidate/probe/smoke not contained in the original training split")
    if original & validation or (original | validation) & final:
        raise ValueError("training, validation and Final data are not isolated")
    if sha(root / "data/grpo/train.parquet") != manifest["original_train_parquet_sha256"]:
        raise ValueError("original training parquet changed")
    # No row transformation is performed. The restored parquet files are exact
    # authenticated archive bytes; Linux runtime also checks their row IDs.
    for relative in (f"{CANDIDATE}/manifest.json", "data/grpo/validation.parquet",
                     "data/grpo/validation.jsonl", "data/environment.json"):
        hashes[relative] = sha(root / relative)
    code_paths = [path for directory, pattern in (("src", "*.py"), ("scripts", "*.py"),
                  ("configs", "*.yaml"), ("configs", "*.json"), ("patches", "*.patch"))
                  for path in (root / directory).rglob(pattern)]
    code_hashes = {path.relative_to(root).as_posix(): sha(path) for path in sorted(code_paths)}
    return {"schema_version": 1, "overlay_version": VERSION,
            "parent_adapter_sha256": PARENT_SHA, "train_count": 953,
            "probe_task_ids": ids(candidate / "probe60.jsonl"),
            "smoke_task_ids": ids(candidate / "smoke8.jsonl"),
            "validation_count": 50, "group_size": 4, "seed": 42,
            "main_optimizer_updates": 64, "efficiency_coefficient": 0.,
            "method": "grpo", "gigpo_enabled": False, "length_shaping_enabled": False,
            "input_sha256": hashes, "training_data": CANDIDATE + "/train953.parquet",
            "code_sha256": code_hashes,
            "probe_data": CANDIDATE + "/probe60.parquet",
            "smoke_data": CANDIDATE + "/smoke8.parquet",
            "validation_data": "data/grpo/validation.parquet",
            "native_environment_modified": False, "final_evaluation_authorized": False,
            "gpu_queue_created": False,
            "required_order": ["native_validation50_diagnostics", "probe60", "cpu_reward_audit",
                               "one_update_smoke_and_export", "main64"],
            "pending_gpu_verification": ["native inference and weight synchronization",
                "three-model unpadded validation50 receipts", "fresh probe60 coverage",
                "one-update GRPO, complete native checkpoint and CPU export"],
            "local_scope": "fixed input hashes and JSONL isolation; no GPU calls or model loading"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("choose a new plan path; do not overwrite frozen inputs")
    report = prepare()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(args.output), "train_count": report["train_count"],
                      "probe_count": len(report["probe_task_ids"]), "gpu_started": False}))


if __name__ == "__main__":
    main()
