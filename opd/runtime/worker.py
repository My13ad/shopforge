"""Run the authorized OPD pilot in isolated GPU phases with saved receipts."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

from common import now, read, sha, write, runtime_paths


def main():
    parser = argparse.ArgumentParser()
    for name in ["workspace", "shop-root", "adapter", "reference", "calibration", "run"]:
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ('student-model', 'teacher-model', 'student-python', 'teacher-python'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--expected-parent-sha256', help='Explicit digest when using a new SFT reference')
    parser.add_argument('--training-task-manifest', type=Path)
    parser.add_argument('--shop-url', default='http://127.0.0.1:5700')
    parser.add_argument("--preparation-only", action="store_true")
    parser.add_argument("--training-tasks", type=int, default=64)
    parser.add_argument("--tasks-per-update", type=int, default=8)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=20261014)
    parser.add_argument("--task-prefix-run", type=Path)
    parser.add_argument("--timeout-minutes", type=int, default=360)
    parser.add_argument("--keep-optimizer-every", type=int, default=0,
                        help="Keep every Nth and the latest optimizer; zero keeps all")
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.training_tasks <= 0 or args.tasks_per_update <= 0 or args.training_tasks % args.tasks_per_update:
        parser.error("Training task count must be a positive whole number of batches")
    if args.timeout_minutes <= 0 or args.keep_optimizer_every < 0 or args.min_free_gib < 2:
        parser.error("Timeout, retention interval and disk reserve are invalid")
    root, run = args.shop_root.resolve(), args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    paths = runtime_paths(args)
    python = paths['python']
    started = time.monotonic()
    child, server, server_log = None, None, None
    progress_file = None
    state = {"phase": "preparing", "pid": os.getpid(), "started_at": now(),
             "optimizer_updates_completed": 0,
             "expected_optimizer_updates": args.training_tasks // args.tasks_per_update,
             "expected_training_tasks": args.training_tasks, "opd_training_started": False,
             "final_evaluation_started": False, "hard_timeout_minutes": args.timeout_minutes,
             "keep_optimizer_every": args.keep_optimizer_every,
             "min_initial_free_gib": args.min_free_gib}
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8",
               PYTHONPATH=str(args.workspace / "src"))

    def snapshot():
        state.update(elapsed_minutes=round((time.monotonic() - started) / 60, 2), updated_at=now())
        if progress_file is not None and progress_file.exists():
            state["active_progress"] = read(progress_file)
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True, timeout=10)
        used, util = map(int, raw.strip().split(","))
        state.update(gpu_used_mib=used, gpu_utilization=util,
                     peak_gpu_used_mib=max(used, state.get("peak_gpu_used_mib", 0)),
                     free_disk_gib=round(shutil_free(run) / 1024**3, 3))
        with (run / "gpu.csv").open("a", encoding="utf-8") as stream:
            stream.write(f"{now()},{state['phase']},{used},{util}\n")
        write(run / "pipeline-status.json", state)
        if time.monotonic() - started > args.timeout_minutes * 60:
            raise TimeoutError("OPD exceeded its configured hard execution limit")

    def prune_optimizer(path, label):
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(run):
            raise ValueError("Optimizer retention may only remove an owned file inside this run")
        receipt = read(path.with_name("status.json"))
        digest = sha(path)
        assert receipt["optimizer_state_finite"] and receipt["optimizer_sha256"] == digest
        events_path = run / "optimizer-retention-events.json"
        events = read(events_path) if events_path.exists() else []
        event = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size,
                 "optimizer_step_values": receipt["optimizer_step_values"],
                 "reason": label, "deleted_at": now()}
        path.unlink()
        events.append(event)
        write(events_path, events)
        state["pruned_optimizer_files"] = len(events)

    def wait_gpu_release():
        deadline = time.monotonic() + 60
        while True:
            snapshot()
            if state["gpu_used_mib"] < 2000:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("GPU was not released between isolated phases")
            time.sleep(5)

    def execute(name, command, status_path=None, needs_teacher=False):
        nonlocal child, progress_file
        if shutil_free(run) < 1024**3:
            raise RuntimeError("Less than 1GiB free disk; refusing to start another phase")
        state.update(phase=name, active_progress=None, active_command=command)
        progress_file = status_path
        write(run / f"{name}-command.json", {"argv": command})
        snapshot()
        with (run / f"{name}.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, env=env, cwd=args.workspace, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state["active_pid"] = child.pid
            print("PHASE_STARTED", name, child.pid, flush=True)
            while child.poll() is None:
                if needs_teacher and server.poll() is not None:
                    raise RuntimeError("Owned teacher server exited during scoring")
                snapshot()
                time.sleep(10)
        state["last_child_exit_code"] = child.returncode
        if child.returncode:
            raise RuntimeError(f"{name} exited {child.returncode}; see preserved log")
        print("PHASE_COMPLETED", name, flush=True)
        snapshot()

    def start_teacher(label):
        nonlocal server, server_log
        wait_gpu_release()
        # Refuse to reuse an unrelated process on the reserved local port.
        try:
            with urlopen("http://127.0.0.1:5800/health", timeout=2):
                raise RuntimeError("Teacher port is already occupied")
        except RuntimeError:
            raise
        except Exception:
            pass
        argv = [str(paths['teacher_python']), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(paths['teacher']), "--served-model-name", "qwen38-teacher",
            "--host", "127.0.0.1", "--port", "5800", "--dtype", "bfloat16", "--max-model-len", "24576",
            "--gpu-memory-utilization", "0.85", "--max-num-seqs", "1", "--max-num-batched-tokens", "1024",
            "--enforce-eager", "--no-enable-prefix-caching", "--limit-mm-per-prompt", '{"image":0,"video":0}',
            "--no-enable-log-requests"]
        write(run / f"{label}-teacher-server-config.json", {"argv": argv, "same_inference_budget_as_calibrated_teacher": True})
        server_log = (run / f"{label}-teacher-server.log").open("w", encoding="utf-8")
        server = subprocess.Popen(argv, env=dict(env, VLLM_NO_USAGE_STATS="1"), stdout=server_log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        state.update(phase=label + "_starting_teacher", teacher_server_pid=server.pid)
        ready_deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError("Teacher failed before readiness")
            try:
                with urlopen("http://127.0.0.1:5800/health", timeout=5) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            if time.monotonic() > ready_deadline:
                raise TimeoutError("Teacher startup exceeded 15 minutes")
            snapshot()
            time.sleep(10)

    def stop_teacher():
        nonlocal server, server_log
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=60)
        if server_log is not None:
            server_log.close()
        server, server_log = None, None
        state["teacher_server_pid"] = None
        wait_gpu_release()

    try:
        snapshot()
        if state["gpu_used_mib"] >= 2000:
            raise RuntimeError("GPU is occupied before the pilot")
        if shutil_free(run) < args.min_free_gib * 1024**3:
            raise RuntimeError("Insufficient free disk for the requested initial reserve")
        execute("objective_tests", [str(python), str(Path(__file__).with_name("test_objectives.py"))])
        execute("preparation", [str(python), str(Path(__file__).with_name("phases.py")), "prepare",
            "--run", str(run), "--workspace", str(args.workspace), "--model", str(paths['model']),
            "--teacher", str(paths['teacher']), "--adapter", str(args.adapter),
            "--reference", str(args.reference), "--calibration", str(args.calibration),
            "--training-tasks", str(args.training_tasks), "--tasks-per-update", str(args.tasks_per_update),
            "--sampling-seed", str(args.sampling_seed), "--selection-seed", str(args.selection_seed)]
            + (["--task-prefix-run", str(args.task_prefix_run)] if args.task_prefix_run else [])
            + (["--expected-parent-sha256", args.expected_parent_sha256] if args.expected_parent_sha256 else [])
            + (["--training-task-manifest", str(args.training_task_manifest)] if args.training_task_manifest else []),
            run / "preparation-status.json")
        if args.preparation_only:
            state.update(phase="preparation_completed", finished_at=now())
            return
        config = read(run / "run_config.json")
        script = run / "source/opd-pilot/phases.py"
        prefix = [str(python), "-u", str(script)]
        adapter = args.adapter
        optimizer_state = None
        # One disposable update first; the actual pilot restarts from exact E1.
        for update in [-1, *range(config["optimizer_steps"])]:
            smoke = update == -1
            number = 0 if smoke else update
            label = "smoke" if smoke else f"round-{update + 1:02d}"
            current = run / label
            current.mkdir()
            wait_gpu_release()
            execute(label + "_rollout", prefix + ["rollout", "--run", str(run), "--adapter", str(adapter),
                "--policy-sha", sha(adapter / "adapter_model.safetensors"), "--update", str(number),
                "--output", str(current / "rollout"), "--shop-url", args.shop_url] + (["--smoke"] if smoke else []), current / "rollout/status.json")
            start_teacher(label)
            execute(label + "_scoring", prefix + ["score", "--run", str(run), "--input", str(current / "rollout"),
                "--output", str(current / "scoring"), "--update", str(number)], current / "scoring/status.json", needs_teacher=True)
            stop_teacher()
            command = prefix + ["train", "--run", str(run), "--adapter", str(adapter),
                "--input", str(current / "scoring"), "--output", str(current / "training"), "--update", str(number)]
            if smoke:
                command.append("--smoke")
            elif optimizer_state:
                command += ["--optimizer-state", str(optimizer_state)]
            state["opd_training_started"] = True
            execute(label + "_training", command, current / "training/status.json")
            receipt = read(current / "training/status.json")
            assert receipt["phase"] == "completed" and receipt["gradient_and_adapter_finite"]
            assert receipt["optimizer_step_values"] == [number + 1] and receipt["optimizer_state_finite"]
            if smoke:
                state["smoke_passed"] = True
                state["smoke_adapter_discarded_from_pilot"] = True
                if args.keep_optimizer_every:
                    prune_optimizer(current / "training/optimizer.pt", "Disposable smoke update; not used to initialize training")
            else:
                if optimizer_state and args.keep_optimizer_every and update % args.keep_optimizer_every:
                    prune_optimizer(optimizer_state, "Successfully loaded into and superseded by the next finite update")
                adapter = current / "training/adapter"
                optimizer_state = current / "training/optimizer.pt"
                state["optimizer_updates_completed"] = update + 1
            snapshot()
        state["opd_adapter"] = str(adapter)
        wait_gpu_release()
        execute("sft_control_training", prefix + ["train", "--run", str(run), "--adapter", str(args.adapter),
            "--control", "--output", str(run / "control-training")], run / "control-training/status.json")
        state["control_adapter"] = str(run / "control-training/adapter")
        (run / "evaluation").mkdir()
        for name, candidate in [("control", run / "control-training/adapter"), ("opd", adapter)]:
            wait_gpu_release()
            execute(name + "_development_evaluation", [str(python), "-u", str(run / "source/evaluate_candidate.py"),
                "--workspace", str(args.workspace), "--model", str(paths['model']), "--adapter", str(candidate),
                "--reference-evaluation", str(args.reference), "--output", str(run / "evaluation" / name),
                "--count", "200", "--base-url", args.shop_url], run / "evaluation" / name / "status.json")
        execute("result_comparison", prefix + ["analyse", "--run", str(run)])
        state.update(phase="completed", finished_at=now(), comparison_file=str(run / "comparison.json"))
        state.update(active_pid=None, active_command=None)
        progress_file = None
        wait_gpu_release()
    except Exception as error:
        state.update(phase="failed", error=f"{type(error).__name__}: {error}", finished_at=now())
        write(run / "pipeline-status.json", state)
        raise
    finally:
        # Clean up only the children explicitly owned by this invocation.
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                state["child_cleanup"] = "SIGINT requested; child still running"
        if server is not None:
            try:
                stop_teacher()
            except Exception as error:
                state["teacher_cleanup_error"] = str(error)
        write(run / "pipeline-status.json", state)


def shutil_free(path):
    import shutil
    return shutil.disk_usage(path).free


if __name__ == "__main__":
    main()
