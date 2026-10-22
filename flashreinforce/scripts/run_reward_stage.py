"""Launch one explicit reward-experiment stage after verifying its CPU receipts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_reward_experiment import ROOT, PARENT_SHA, prepare, sha


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_prerequisites(args, plan):
    """Fail closed; this entry never queues another GPU job after this stage."""
    current = prepare()
    if current != plan:
        raise ValueError("frozen reward plan no longer matches current inputs")
    adapter = args.adapter.resolve()
    if sha(adapter / "adapter_model.safetensors") != PARENT_SHA:
        raise ValueError("all new reward stages must start independently from original OPD-32")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("choose a new empty output directory")
    if args.checkpoint_root and args.checkpoint_root.exists() and any(args.checkpoint_root.iterdir()):
        raise ValueError("checkpoint root must be new or empty")
    if args.stage == "probe":
        return
    if not args.probe_audit or not args.probe_receipts:
        raise ValueError("smoke/main require the fresh probe audit AND its original receipts")
    from shopping_grpo.training.grpo.reward_audit import audit_probe
    records = [read(path) for path in args.probe_receipts.glob("*.json")]
    audit = audit_probe(records, expected_tasks=set(plan["probe_task_ids"]), parent_sha=PARENT_SHA)
    saved = read(args.probe_audit)
    if (not audit["ready_for_smoke"] or saved.get("plan_sha256") != sha(args.plan)
            or any(saved.get(key) != value for key, value in audit.items())):
        raise ValueError("probe evidence/audit failed the unchanged gates or belongs to another plan")
    if args.stage == "main":
        if not args.diagnostics or not args.smoke_verification:
            raise ValueError("main requires native diagnostics and a verified one-update checkpoint export")
        diagnostic = read(args.diagnostics)
        if (not diagnostic.get("no_training_updates") or not diagnostic.get("no_final200")
                or set(diagnostic.get("arms", {})) != {"opd32", "r0_64", "r1_64"}
                or any(arm.get("tasks") != 50 for arm in diagnostic["arms"].values())):
            raise ValueError("complete original three-model validation50 diagnostics are required")
        # Recheck raw post-unpad task records and adapter identities, not a flag alone.
        from shopping_grpo.training.grpo.diagnostics import validate_task_records
        expected_adapters = read(ROOT / "outputs/preparation/v1/diagnostic-adapters.json")
        expected_tasks = {int(json.loads(line)["task_id"]) for line in
                          (ROOT / "data/grpo/validation.jsonl").read_text().splitlines() if line.strip()}
        for arm, adapter_sha in expected_adapters.items():
            receipts = [read(p) for p in (args.diagnostics.parent / arm / "task_rollouts").glob("*.json")]
            validate_task_records(receipts, expected_tasks, adapter_sha)
            checks = diagnostic.get("runtime_audits", {}).get(arm, {})
            if not all(checks.get(key) is True for key in
                       ("actor_adapter_verified", "sender_receiver_samples_match", "readback_verified")):
                raise ValueError("runtime weight identity audit incomplete: " + arm)
        smoke = read(args.smoke_verification)
        if (smoke.get("optimizer_updates") != 1 or smoke.get("parent_adapter_sha256") != PARENT_SHA
                or smoke.get("plan_sha256") != sha(args.plan) or not smoke.get("finite_adapter")
                or not smoke.get("native_checkpoint_complete") or not smoke.get("changed_parameter_elements", 0)):
            raise ValueError("one-update export verification does not match this reward experiment")
        for relative, expected_sha in smoke["verified_file_sha256"].items():
            if sha(args.smoke_verification.parent / relative) != expected_sha:
                raise ValueError("smoke checkpoint/export file changed: " + relative)


def command(args, plan):
    profile = {"probe": "probe", "smoke": "smoke", "main": "main"}[args.stage]
    train = plan["smoke_data"] if args.stage == "smoke" else plan["training_data"]
    validation = plan["probe_data"] if args.stage == "probe" else plan["validation_data"]
    launch = [sys.executable, "scripts/train_grpo.py", "--model", str(args.model.resolve()),
            "--adapter", str(args.adapter.resolve()), "--method", "grpo", "--profile", profile,
            "--reward-overlay", plan["overlay_version"], "--seed", "42",
            "--train-data", str(ROOT / train), "--val-data", str(ROOT / validation),
            "--output", str(args.output.resolve()), "--env-url", args.env_url,
            "--experiment-name", args.output.name, "--",
            "data.filter_overlong_prompts_workers=4",
            "actor_rollout_ref.actor.calculate_entropy=false",
            "actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=true"]
    if args.checkpoint_root:
        launch.append("trainer.default_local_dir=" + json.dumps(str(args.checkpoint_root.resolve())))
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("probe", "smoke", "main"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, help="optional verified checkpoint filesystem, e.g. RAM")
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--probe-audit", type=Path)
    parser.add_argument("--probe-receipts", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--smoke-verification", type=Path)
    parser.add_argument("--execute", action="store_true", help="otherwise only validate/print the command")
    args = parser.parse_args()
    plan = read(args.plan)
    check_prerequisites(args, plan)
    launch = command(args, plan)
    print(json.dumps({"command": launch, "executing": args.execute, "next_stage_queued": False}))
    if not args.execute:
        return
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    # Entrypoint performs the pinned runtime and full parquet isolation preflight.
    raise SystemExit(subprocess.call(launch, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
