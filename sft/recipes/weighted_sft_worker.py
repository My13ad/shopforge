"""Execute one segment-weighted SFT run with a frozen source/data snapshot."""

import argparse
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument('--loss-mode', choices=('smartad', 'option-weighted'), required=True)
    args = parser.parse_args()
    workspace, model, run = args.workspace.resolve(), args.model.resolve(), args.run.resolve()
    if not (model / "config.json").is_file():
        raise SystemExit("Missing model config")
    if shutil.disk_usage(workspace).free < 10 * 1024**3:
        raise SystemExit("Less than 10 GiB free disk space")
    run.mkdir(parents=True, exist_ok=False)
    status = {"phase": "preparing", "started_at": timestamp(), "worker_pid": os.getpid(),
              "run_directory": str(run), "global_step": 0, "total_steps": None,
              "peak_total_gpu_memory_mib": 0}
    save_json(run / "status.json", status)
    started = time.monotonic()
    try:
        import hashlib
        sys.path.insert(0, str(workspace / "src"))
        from shopping_grpo.evaluation.blind_guard import guard_blind_final
        original_paths = [workspace / "data/sft/train.jsonl", workspace / "data/sft/validation.jsonl"]
        guard_blind_final(original_paths, allowed=False)
        task_sets, data_metadata = [], {}
        (run / "input").mkdir()
        for path in original_paths:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            ids = {row["task_id"] for row in rows}
            assert len(ids) == len(rows)
            task_sets.append(ids)
            data_metadata[path.stem] = {"rows": len(rows), "unique_tasks": len(ids),
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            shutil.copy2(path, run / "input" / path.name)
        assert len(task_sets[0]) == 800 and len(task_sets[1]) == 200
        assert not (task_sets[0] & task_sets[1])
        (run / "source/scripts").mkdir(parents=True)
        shutil.copytree(workspace / "src", run / "source/src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(workspace / "scripts/train_lora_sft.py", run / "source/scripts/train_lora_sft.py")
        command = [
            sys.executable, "-u", str(run / "source/scripts/train_lora_sft.py"),
            "--model", str(model), "--train", str(run / "input/train.jsonl"),
            "--validation", str(run / "input/validation.jsonl"),
            "--output", str(run / "adapter"), "--max-length", "28672", "--epochs", "3",
            "--dtype", "bf16", "--per-device-train-batch-size", "1",
            "--per-device-eval-batch-size", "1", "--gradient-accumulation-steps", "8",
            "--learning-rate", "1e-4", "--warmup-ratio", "0.03",
            "--lora-r", "16", "--lora-alpha", "32", "--lora-dropout", "0.05",
            "--gradient-checkpointing", "--attention-implementation", "sdpa", "--liger-kernel",
            "--logging-steps", "1", "--save-total-limit", "3", "--seed", "42",
            '--loss-mode', args.loss_mode, '--linear-attention-backend', 'torch',
        ]
        packages = {name: importlib.metadata.version(name)
                    for name in ["torch", "transformers", "peft", "accelerate", "liger-kernel"]}
        save_json(run / "run_config.json", {
            "command": command, "data": data_metadata, "packages": packages,
            "python": sys.version, "model": str(model), "method": args.loss_mode,
            "training_script_sha256": hashlib.sha256(Path(command[2]).read_bytes()).hexdigest(),
        })
        env = dict(os.environ, PYTHONPATH=str(run / "source/src"), HF_HUB_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8")
        with (run / "train.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=workspace, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            status.update(phase="training", trainer_pid=process.pid)
            save_json(run / "status.json", status)
            print("TRAINER_STARTED", process.pid, run, flush=True)
            with (run / "gpu.csv").open("w", encoding="utf-8", buffering=1) as gpu_log:
                gpu_log.write("timestamp,memory_used_mib,utilization_percent\n")
                while process.poll() is None:
                    try:
                        raw = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], text=True, timeout=5,
                        )
                        memory, utilization = [int(value.strip()) for value in raw.splitlines()[0].split(",")]
                        gpu_log.write(f"{timestamp()},{memory},{utilization}\n")
                        status["peak_total_gpu_memory_mib"] = max(status["peak_total_gpu_memory_mib"], memory)
                        status["gpu_memory_used_mib"] = memory
                        status["gpu_utilization_percent"] = utilization
                    except (ValueError, subprocess.SubprocessError):
                        pass
                    with (run / "train.log").open("rb") as reader:
                        reader.seek(max(0, reader.seek(0, 2) - 32000))
                        tail = reader.read().decode("utf-8", errors="replace")
                    steps = re.findall(r"\[step (\d+)/(\d+)\]", tail)
                    if steps:
                        status["global_step"], status["total_steps"] = map(int, steps[-1])
                    status["elapsed_minutes"] = round((time.monotonic() - started) / 60, 1)
                    status["updated_at"] = timestamp()
                    save_json(run / "status.json", status)
                    time.sleep(15)
        status["exit_code"] = process.returncode
        if process.returncode:
            status["phase"] = "failed"
        else:
            summary = json.loads((run / "adapter/train_summary.json").read_text(encoding="utf-8"))
            assert (run / "adapter/adapter_model.safetensors").is_file()
            assert summary['train_examples'] == 800 and summary['validation_examples'] == 200
            assert summary['supervision']['mode'] == args.loss_mode
            assert summary['acceleration']['linear_attention_backend'] == 'torch'
            evaluations = [entry for entry in summary["log_history"] if "eval_loss" in entry]
            assert math.isfinite(summary["train_loss"])
            assert len(evaluations) == 3 and all(math.isfinite(entry["eval_loss"]) for entry in evaluations)
            assert max(entry['step'] for entry in evaluations) == 300
            from safetensors.torch import load_file
            import torch
            tensors = load_file(str(run / 'adapter/adapter_model.safetensors'))
            assert tensors and all(torch.isfinite(value).all().item() for value in tensors.values())
            b_matrices = [value for key, value in tensors.items() if 'lora_B' in key]
            assert b_matrices and all(value.abs().max().item() > 0 for value in b_matrices)
            save_json(run / 'adapter-audit.json', {
                'passed': True, 'tensor_count': len(tensors), 'updated_b_matrices': len(b_matrices),
                'sha256': hashlib.sha256((run / 'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),
            })
            best = min(evaluations, key=lambda entry: entry["eval_loss"])
            best_path = run / "adapter" / f"checkpoint-{best['step']}"
            save_json(run / "completion_summary.json", {
                "train_examples": summary["train_examples"],
                "validation_examples": summary["validation_examples"],
                "train_loss": summary["train_loss"], "evaluations": evaluations,
                "best_validation_checkpoint": str(best_path) if best_path.exists() else None,
                "final_adapter": str(run / "adapter"),
                "peak_torch_memory_gib": summary["peak_gpu_memory_gib"],
                "peak_total_gpu_memory_mib": status["peak_total_gpu_memory_mib"],
            })
            status.update(phase="completed", global_step=max(entry["step"] for entry in evaluations))
        status.update(finished_at=timestamp(), elapsed_minutes=round((time.monotonic() - started) / 60, 1))
        save_json(run / "status.json", status)
        print("TRAINER_FINISHED", status["phase"], process.returncode, flush=True)
        if status['phase'] != 'completed':
            raise SystemExit(process.returncode or 1)
    except Exception as error:
        status.update(phase="failed", error=f"{type(error).__name__}: {error}", finished_at=timestamp())
        save_json(run / "status.json", status)
        raise


if __name__ == "__main__":
    main()
