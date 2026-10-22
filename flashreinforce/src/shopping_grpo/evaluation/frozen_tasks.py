"""Verify a separately frozen confirmation set without changing validation50."""
import hashlib
import json
from pathlib import Path

PROTOCOL = "shopping-unseen-confirmation-v1"


def validate_frozen_evaluation(manifest_path, parquet_path, *, root):
    import pyarrow.parquet as pq
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if set(manifest) != {"protocol", "task_ids", "parquet_sha256", "exclusion_audit_sha256", "seed"} or manifest["protocol"] != PROTOCOL:
        raise ValueError("invalid frozen confirmation manifest")
    if hashlib.sha256(Path(parquet_path).read_bytes()).hexdigest() != manifest["parquet_sha256"]:
        raise ValueError("confirmation parquet fingerprint differs")
    audit_file = path.with_name("exclusion-audit.json")
    if hashlib.sha256(audit_file.read_bytes()).hexdigest() != manifest["exclusion_audit_sha256"]:
        raise ValueError("confirmation exclusion audit fingerprint differs")
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    excluded = set(audit["excluded_task_ids"])
    for rel in ("data/grpo/train.jsonl", "data/grpo/validation.jsonl", "data/evaluation/tasks.jsonl"):
        frozen = {json.loads(s)["task_id"] for s in (Path(root) / rel).read_text(encoding="utf-8").splitlines() if s.strip()}
        if not frozen <= excluded:
            raise ValueError("confirmation audit omits frozen train/validation/Final tasks")
    actual = [int(r["extra_info"]["task_id"]) for r in pq.read_table(parquet_path, columns=["extra_info"]).to_pylist()]
    if actual != manifest["task_ids"] or len(actual) != 200 or len(set(actual)) != 200 or excluded.intersection(actual):
        raise ValueError("confirmation task identity, denominator or isolation differs")
    return manifest
