"""Build a small allowlisted code/data update with a per-file SHA256 manifest."""
import argparse
import json
from pathlib import Path
import zipfile

from prepare_reward_experiment import ROOT, CANDIDATE, prepare, sha


def build(output, plan_path, checks_path):
    if output.exists():
        raise ValueError("preserve the previous bundle; choose a new filename")
    if json.loads(plan_path.read_text(encoding="utf-8")) != prepare():
        raise ValueError("plan no longer matches the current code/data")
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    if checks.get("pytest_exit_code") != 0 or checks.get("gpu_started") is not False:
        raise ValueError("a successful CPU-only test receipt is required")
    files = {path for folder, pattern in (("src", "*.py"), ("scripts", "*.py"),
        ("tests", "*.py"), ("configs", "*.yaml"), ("configs", "*.json"), ("patches", "*.patch"))
        for path in (ROOT/folder).rglob(pattern)}
    files.update((ROOT/relative) for relative in (
        "AGENTS.md", "pyproject.toml", "docs/reward-experiment.md",
        "data/environment.json", "data/grpo/train.jsonl", "data/grpo/train.parquet",
        "data/grpo/validation.jsonl", "data/grpo/validation.parquet", "data/evaluation/tasks.jsonl",
        "outputs/preparation/v1/diagnostic-adapters.json", "outputs/preparation/v1/post-r1-cpu-audit.json",
        "outputs/preparation/v1/run_native_diagnostics.py", "outputs/preparation/v1/analyze_native_diagnostics.py",
        "outputs/preparation/v1/deadline_watchdog.py", "outputs/preparation/v1/run_bounded_stage.py",
        "outputs/preparation/v1/persist_native_checkpoint.py", "outputs/preparation/v1/persist_stage_checkpoints.py",
        "outputs/preparation/v1/runtime-wheel/source/verl/trainer/ppo/ray_trainer.py",
        "outputs/preparation/v1/runtime-wheel/source/verl/protocol.py",
        "outputs/preparation/reward-v1/data-audit.json"))
    files.update(path for path in (ROOT/CANDIDATE).iterdir() if path.is_file())
    files.update((plan_path, checks_path))
    manifest_path = ROOT / "outputs/preparation/reward-v1/source-manifest.json"
    manifest = {"schema_version": 2, "scope": "CPU-prepared spec reward and native validation-unpad update",
        "gpu_started": False, "automatic_gpu_queue_created": False,
        "files": {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(files)}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files.add(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        import hashlib
        for relative, expected in manifest["files"].items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                raise ValueError("archive readback hash mismatch: " + relative)
    receipt = {"archive": output.name, "archive_sha256": sha(output), "bytes": output.stat().st_size,
               "file_count": len(files), "readback_verified": True, "gpu_started": False,
               "source_manifest_sha256": sha(manifest_path)}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve(), args.plan.resolve(), args.checks.resolve())))


if __name__ == "__main__":
    main()
