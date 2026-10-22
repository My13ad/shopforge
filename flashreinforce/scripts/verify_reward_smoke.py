"""CPU proof of one GRPO update, native checkpoint, optimizer and LoRA export."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from prepare_reward_experiment import ROOT, PARENT_SHA, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="smoke train_grpo output directory")
    parser.add_argument("--checkpoint-root", type=Path, help="native checkpoint root if it differs from --run")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.export.exists():
        raise SystemExit("preserve previous smoke receipts and exports; choose new paths")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    launch = json.loads((args.run / "launch.json").read_text(encoding="utf-8"))
    if (launch.get("profile") != "smoke" or launch.get("method") != "grpo"
            or launch.get("training_reward_overlay") != plan["overlay_version"]
            or launch.get("parent_adapter_sha256") != PARENT_SHA
            or sha(args.parent_adapter / "adapter_model.safetensors") != PARENT_SHA):
        raise SystemExit("smoke was not the fixed original-OPD32 GRPO reward experiment")
    events = [json.loads(line) for line in (args.run / "training_diagnostics.jsonl").read_text().splitlines() if line.strip()]
    updates = [event for event in events if event["event"] == "optimizer_step"]
    if len(updates) != 1 or updates[0]["global_step"] != 1:
        raise SystemExit("smoke requires exactly one committed optimizer update")
    metrics = updates[0]["metrics"]
    grad = float(metrics["actor/grad_norm"])
    if metrics["training/optimizer_updated"] != 1 or not math.isfinite(grad) or grad <= 0:
        raise SystemExit("smoke optimizer/gradient verification failed")
    generated = [info for event in events if event["event"] == "generation_batch"
                 for info in event.get("rollouts", [])]
    if not generated or any(info.get("training_reward_overlay", {}).get("version") != plan["overlay_version"]
                            for info in generated):
        raise SystemExit("smoke lacks versioned training overlay evidence")
    checkpoint_root = args.checkpoint_root or args.run
    source = checkpoint_root / "global_step_1"
    if (checkpoint_root / "latest_checkpointed_iteration.txt").read_text().strip() != "1":
        raise SystemExit("native checkpoint iteration differs from the committed update")
    native_files = [source / name for name in ("data.pt", "actor/model_world_size_1_rank_0.pt",
        "actor/optim_world_size_1_rank_0.pt", "actor/extra_state_world_size_1_rank_0.pt",
        "actor/fsdp_config.json", "actor/lora_train_meta.json")]
    if any(not path.is_file() or not path.stat().st_size for path in native_files):
        raise SystemExit("native smoke checkpoint is incomplete")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"))
    command = [sys.executable, "scripts/export_grpo_adapter.py", "--checkpoint", str(source / "actor"),
               "--parent-adapter", str(args.parent_adapter), "--output", str(args.export)]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    import torch
    from safetensors import safe_open
    changed, count = 0, 0
    with safe_open(args.parent_adapter / "adapter_model.safetensors", framework="pt", device="cpu") as old, \
            safe_open(args.export / "adapter_model.safetensors", framework="pt", device="cpu") as new:
        if set(old.keys()) != set(new.keys()):
            raise SystemExit("exported adapter tensor names differ from parent")
        for key in old.keys():
            left, right = old.get_tensor(key).float(), new.get_tensor(key).float()
            if left.shape != right.shape or not torch.isfinite(right).all():
                raise SystemExit("invalid exported adapter tensor: " + key)
            changed += int(torch.count_nonzero(left-right))
            count += 1
    if count != 372 or not changed:
        raise SystemExit("smoke adapter has no verified update or incomplete tensor coverage")
    initialized = False
    with tempfile.TemporaryDirectory() as temporary:
        try:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group("gloo", init_method=(Path(temporary)/"group").as_uri(),
                                                     rank=0, world_size=1)
                initialized = True
            optimizer = torch.load(source / "actor/optim_world_size_1_rank_0.pt", map_location="cpu", weights_only=False)
            steps = []
            for state in optimizer["state"].values():
                if "step" in state:
                    value = state["step"]
                    value = value.to_local() if hasattr(value, "to_local") else value
                    steps.append(float(value))
            if not steps or min(steps) != 1 or max(steps) != 1:
                raise SystemExit("optimizer state does not contain exactly one update")
        finally:
            if initialized:
                torch.distributed.destroy_process_group()
    receipt = json.loads((args.export / "export-manifest.json").read_text(encoding="utf-8"))
    for name, expected in receipt["files"].items():
        if sha(args.export / name) != expected:
            raise SystemExit("export file hash mismatch: " + name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = native_files + [path for path in args.export.iterdir() if path.is_file()] + [args.run / "launch.json"]
    report = {"schema_version": 1, "optimizer_updates": 1, "parent_adapter_sha256": PARENT_SHA,
              "plan_sha256": sha(args.plan), "overlay_version": plan["overlay_version"],
              "finite_adapter": True, "native_checkpoint_complete": True,
              "adapter_tensors": count, "changed_parameter_elements": changed,
              "optimizer_step_min": min(steps), "optimizer_step_max": max(steps),
              "actor_grad_norm": grad, "native_checkpoint_preserved": True,
              "overlay_applied_rollouts": sum(bool(info["training_reward_overlay"].get("applied"))
                                               for info in generated),
              "verified_file_sha256": {os.path.relpath(path.resolve(), args.output.parent.resolve()): sha(path) for path in files}}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"optimizer_updates": 1, "changed_parameter_elements": changed, "verified": True}))


if __name__ == "__main__":
    main()
