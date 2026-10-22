#!/usr/bin/env python3
"""Run a controlled Shop GRPO, GiGPO or FlashREINFORCE comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from shopping_grpo.training.grpo.gigpo import SCHEMA_VERSION as GIGPO_SCHEMA_VERSION
from shopping_grpo.training.grpo.reward_overlay import MODES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/grpo.yaml"
DEFAULT_AGENT_CONFIG = ROOT / "configs/agent_loop.yaml"
DEFAULT_TOOL_CONFIG = ROOT / "configs/tools.json"
DEFAULT_MANIFEST = ROOT / "data/environment.json"
DEFAULT_MODEL = ROOT / "outputs/models/sft-merged"
DEFAULT_TRAIN_DATA = ROOT / "data/grpo/train.parquet"
DEFAULT_VAL_DATA = ROOT / "data/grpo/validation.parquet"
FLASH_DATA = ROOT / "outputs/preparation/v1/candidate-price953"


def hydra_string(value):
    # Hydra unescapes backslashes but does not decode JSON Unicode escapes.
    return json.dumps(value, ensure_ascii=False)


def _model_has_weights(path: Path) -> bool:
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).is_file() for name in candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, help="Continue a saved LoRA on the unchanged base")
    parser.add_argument("--method", choices=("grpo", "gigpo", "flashreinforce"), default="grpo")
    parser.add_argument("--flash-batch-size", type=int, default=8,
                        help="Independent prompts per FlashREINFORCE update (default: 8)")
    parser.add_argument("--flash-execution", choices=("sync", "async"), default="sync",
                        help="FlashREINFORCE execution: synchronous batches or persistent completion queue")
    parser.add_argument("--flash-invalid-batch-policy", choices=("error", "skip_batch"), default="skip_batch",
                        help="Whole-batch native-reward admission for FlashREINFORCE (default: skip_batch)")
    parser.add_argument("--reward-overlay", choices=sorted(MODES), default="native")
    parser.add_argument("--profile", choices=("custom", "smoke", "probe", "main", "eval"), default="custom")
    parser.add_argument("--identity-contexts", type=Path, help="Frozen training contexts for evaluation identity audit")
    parser.add_argument("--evaluation-task-manifest", type=Path, help="Separately frozen confirmation tasks; eval only")
    parser.add_argument("--checkpoint-selection", action="store_true",
                        help="Select and stop on fixed serial greedy validation50; FlashREINFORCE main only")
    parser.add_argument("--validation-interval", type=int, default=64)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-minimum-updates", type=int, default=128)
    parser.add_argument("--feedback-manifest", type=Path, help="Frozen train-only hard-task ID manifest")
    parser.add_argument("--hard-task-fraction", type=float, default=0.2,
                        help="Hard-lane injection fraction; 0 uses the same sampler as a uniform control")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--output", type=Path, default=Path("outputs/models/grpo"))
    parser.add_argument(
        "--logger",
        choices=("console", "swanlab"),
        default="console",
    )
    parser.add_argument("--experiment-name", default="shopping-agent-grpo")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="additional veRL Hydra overrides after --",
    )
    return parser.parse_args()


def _validated_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    return resolved


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    flash = args.method == "flashreinforce"
    asynchronous = args.flash_execution == "async"
    selection = getattr(args, "checkpoint_selection", False)
    feedback = getattr(args, "feedback_manifest", None)
    evaluation_manifest = getattr(args, "evaluation_task_manifest", None)
    if evaluation_manifest and args.profile != "eval":
        raise SystemExit("evaluation task manifest is evaluation-only")
    if feedback:
        if not flash or not asynchronous or args.profile != "main" or not selection:
            raise SystemExit("feedback sampling requires async FlashREINFORCE main with checkpoint selection")
        if not 0 <= args.hard_task_fraction < 1:
            raise SystemExit("hard task fraction must be in [0, 1)")
    if selection:
        if not flash or args.profile != "main" or args.flash_invalid_batch_policy != "skip_batch":
            raise SystemExit("checkpoint selection requires FlashREINFORCE main with skip_batch admission")
        if (args.validation_interval < 1 or args.early_stopping_patience < 1
                or args.early_stopping_minimum_updates < 0):
            raise SystemExit("invalid checkpoint validation interval or stopping bounds")
    if asynchronous and not flash:
        raise SystemExit("--flash-execution async requires --method flashreinforce")
    if flash and args.flash_batch_size < 2:
        raise SystemExit("FlashREINFORCE needs at least two independent prompts per update")
    if args.reward_overlay != "native" and (args.method != "grpo" or args.profile == "eval"):
        raise SystemExit("the first reward overlay is training-only GRPO; evaluation uses native reward")
    model = _validated_path(args.model, "model directory")
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"model directory is missing config.json: {model}")
    if not _model_has_weights(model):
        raise SystemExit(
            "model directory has no supported weight file or sharded index: "
            f"{model}"
        )
    adapter = None
    adapter_sha = ""
    if args.adapter:
        adapter = _validated_path(args.adapter, "parent adapter")
        weights = adapter / "adapter_model.safetensors"
        if not weights.is_file() or not (adapter / "adapter_config.json").is_file():
            raise SystemExit("parent adapter is missing weights or adapter_config.json")
        if not (adapter / "chat_template.jinja").is_file():
            raise SystemExit("parent adapter must retain its SFT chat_template.jinja")
        with weights.open("rb") as handle:
            adapter_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        settings = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        if settings.get("r") != 16 or settings.get("lora_alpha") != 32:
            raise SystemExit("parent LoRA must have rank=16 and alpha=32")
    if args.profile != "custom" and adapter is None:
        raise SystemExit("controlled profiles require an explicit --adapter parent")
    train_path, val_path = args.train_data, args.val_data
    if flash and train_path == DEFAULT_TRAIN_DATA:
        train_path = FLASH_DATA / ("smoke8.parquet" if args.profile == "smoke" and not asynchronous else "train953.parquet")
    if args.profile == "smoke" and train_path == DEFAULT_TRAIN_DATA:
        train_path = ROOT / "data/grpo/smoke8.parquet"
    if args.profile == "probe":
        if args.method not in ("grpo", "flashreinforce"):
            raise SystemExit("probe supports GRPO or single-rollout FlashREINFORCE without actor updates")
        if val_path == DEFAULT_VAL_DATA:
            val_path = FLASH_DATA / "probe60.parquet" if flash else ROOT / "data/grpo/probe32.parquet"
    train_data = _validated_path(train_path, "train parquet")
    val_data = _validated_path(val_path, "validation/probe parquet")
    config = _validated_path(args.config, "GRPO example config")
    if evaluation_manifest:
        from shopping_grpo.evaluation.frozen_tasks import validate_frozen_evaluation
        evaluation_manifest = _validated_path(evaluation_manifest, "frozen evaluation manifest")
        validate_frozen_evaluation(evaluation_manifest, val_data, root=ROOT)
    if feedback:
        from shopping_grpo.training.grpo.feedback_sampling import load_manifest
        feedback = _validated_path(feedback, "feedback sampling manifest")
        load_manifest(feedback, train_data, root=ROOT)
    output_path = args.output
    if flash and output_path == Path("outputs/models/grpo"):
        output_path = Path("outputs/models/flashreinforce")
    output = output_path.expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise SystemExit(f"output must be a directory: {output}")
        if any(output.iterdir()):
            raise SystemExit(f"output directory must be new or empty: {output}")
    if args.logger == "swanlab" and not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("--logger swanlab requires SWANLAB_API_KEY")

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "SHOPPING_GRPO_ROOT": str(ROOT),
            "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
            "SHOPPING_ENV_MANIFEST": str(DEFAULT_MANIFEST),
            "GRPO_MODEL_PATH": str(model),
            "GRPO_TRAIN_FILE": str(train_data),
            "GRPO_VAL_FILE": str(val_data),
            "GRPO_OUTPUT_DIR": str(output),
            "SHOPPING_GRPO_DIAGNOSTICS_PATH": str(
                output / "training_diagnostics.jsonl"
            ),
            "SHOPSIM_BASE_URL": str(args.env_url),
            "SHOPPING_AGENT_LOOP_CONFIG": str(DEFAULT_AGENT_CONFIG),
            "SHOPPING_TOOL_CONFIG": str(DEFAULT_TOOL_CONFIG),
            "GRPO_CONFIG_NAME": config.stem,
            "SHOPPING_PARENT_ADAPTER_SHA256": adapter_sha,
            "SHOPPING_TRAINING_REWARD_OVERLAY": args.reward_overlay,
            "SHOPPING_DIAGNOSTIC_DIR": "",
            "SHOPPING_DIAGNOSTIC_ROLLOUT_DIR": "",
            "SHOPPING_VALIDATION_RECEIPTS_DIR": str(output / "validation_rollouts")
                if args.profile == "main" else "",
            "SHOPPING_GIGPO_PROBE_DIR": str(output / "probe_rollouts") if args.profile == "probe" else "",
        }
    )
    if args.logger == "swanlab":
        environment.update(
            {
                "SWANLAB_MODE": "online",
                "SWANLAB_LOG_DIR": str(output / "swanlab"),
            }
        )
    if args.profile != "custom" or flash:
        environment["SHOPPING_LENGTH_SHAPING_ENABLE"] = "false"
    logger_override = (
        "trainer.logger=[console,swanlab]"
        if args.logger == "swanlab"
        else "trainer.logger=[console]"
    )
    overrides = [
        logger_override,
        f"trainer.experiment_name={args.experiment_name}",
        f"shopping_gigpo.enable={'true' if args.method == 'gigpo' else 'false'}",
        f"shopping_flashreinforce.enable={'true' if flash else 'false'}",
        f"data.seed={args.seed}",
        f"actor_rollout_ref.actor.data_loader_seed={args.seed}",
        f"+actor_rollout_ref.rollout.engine_kwargs.vllm.seed={args.seed}",
    ]
    if flash:
        overrides += [
            f"shopping_flashreinforce.execution={args.flash_execution}",
            f"shopping_flashreinforce.invalid_batch_policy={args.flash_invalid_batch_policy}",
            f"data.train_batch_size={args.flash_batch_size}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={args.flash_batch_size}",
            "actor_rollout_ref.actor.ppo_epochs=1",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.actor.use_dynamic_bsz=false",
            "actor_rollout_ref.actor.shuffle=false",
            "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
            "actor_rollout_ref.actor.policy_loss._target_="
            "shopping_grpo.training.grpo.flashreinforce_config.FlashPolicyLossConfig",
            "actor_rollout_ref.actor.policy_loss.loss_mode=flashreinforce",
            "+actor_rollout_ref.actor.policy_loss.sequence_kl_threshold="
            "${shopping_flashreinforce.sequence_kl_threshold}",
            "actor_rollout_ref.actor.calculate_entropy=false",
            "actor_rollout_ref.actor.entropy_coeff=0.0",
            "actor_rollout_ref.actor.use_kl_loss=false",
            "actor_rollout_ref.rollout.n=1",
            "actor_rollout_ref.rollout.temperature=1.0",
            "actor_rollout_ref.rollout.top_p=1.0",
            "actor_rollout_ref.rollout.top_k=-1",
            "actor_rollout_ref.rollout.calculate_log_probs=true",
            "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs",
            "++actor_rollout_ref.rollout.repetition_penalty=1.0",
            "algorithm.adv_estimator=flashreinforce",
            "algorithm.use_kl_in_reward=false",
            "algorithm.rollout_correction.bypass_mode=true",
            "algorithm.rollout_correction.rollout_is=null",
            "algorithm.rollout_correction.rollout_rs=null",
            "algorithm.rollout_correction.loss_type=reinforce",
            "shopping_dynamic_sampling.enable=false",
            "shopping_trace.enable=false",
            "critic.enable=false",
            "trainer.critic_warmup=0",
        ]
        if asynchronous:
            overrides += [
                f"shopping_flashreinforce.async_inflight={4 * args.flash_batch_size}",
                "data.dataloader_num_workers=0",
                "actor_rollout_ref.rollout.free_cache_engine=false",
                "actor_rollout_ref.rollout.gpu_memory_utilization=0.25",
                f"actor_rollout_ref.rollout.max_num_seqs={4 * args.flash_batch_size}",
                "+actor_rollout_ref.rollout.checkpoint_manager_class=shopping_grpo.training.grpo.async_runtime.AsyncCheckpointManager",
                "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=shopping_grpo.training.grpo.async_runtime.AsyncAgentLoopManager",
                "actor_rollout_ref.rollout.agent.agent_loop_config_path=" + hydra_string(str(ROOT / "configs/agent_loop_async.yaml")),
            ]
    if adapter:
        overrides.extend([
            f"actor_rollout_ref.model.lora_adapter_path={hydra_string(str(adapter))}",
            f"actor_rollout_ref.model.tokenizer_path={hydra_string(str(adapter))}",
        ])
    if args.profile != "custom":
        overrides += ["trainer.resume_mode=disable", "trainer.max_actor_ckpt_to_keep=1",
                      "trainer.max_critic_ckpt_to_keep=1", "shopping_trace.enable=false"]
    elif flash and args.flash_invalid_batch_policy == "skip_batch":
        overrides.append("trainer.resume_mode=disable")
    if args.profile == "main":
        overrides += ["trainer.total_training_steps=64", "trainer.save_freq=32",
                      "trainer.test_freq=32", "trainer.val_before_train=false"]
    elif args.profile == "smoke":
        overrides += [f"trainer.total_training_steps={4 if asynchronous else 1}",
                      f"trainer.save_freq={4 if asynchronous else 1}",
                      "trainer.test_freq=-1", "trainer.val_before_train=false"]
    elif args.profile == "probe":
        overrides += ["trainer.val_only=true", "trainer.val_before_train=true",
                      f"actor_rollout_ref.rollout.val_kwargs.n={1 if flash else 4}",
                      f"actor_rollout_ref.rollout.val_kwargs.temperature={1.0 if flash else 0.7}",
                      f"actor_rollout_ref.rollout.val_kwargs.top_p={1.0 if flash else 0.9}",
                      "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
                      "trainer.save_freq=-1", "trainer.test_freq=-1",
                      "trainer.total_training_steps=1"]
    overrides += [
        "+ray_kwargs.ray_init.runtime_env.env_vars.SHOPPING_PARENT_ADAPTER_SHA256="
        + hydra_string(adapter_sha),
        "+ray_kwargs.ray_init.runtime_env.env_vars.SHOPPING_GIGPO_PROBE_DIR="
        + hydra_string(environment["SHOPPING_GIGPO_PROBE_DIR"]),
    ]
    for key in ("SHOPPING_TRAINING_REWARD_OVERLAY", "SHOPPING_DIAGNOSTIC_DIR",
                "SHOPPING_DIAGNOSTIC_ROLLOUT_DIR", "SHOPPING_VALIDATION_RECEIPTS_DIR"):
        overrides.append("+ray_kwargs.ray_init.runtime_env.env_vars." + key + "="
                         + hydra_string(environment[key]))
    if flash:
        overrides.append(
            '+ray_kwargs.ray_init.runtime_env.env_vars.SHOPPING_LENGTH_SHAPING_ENABLE="false"'
        )
    for key in ("CUDA_HOME", "CUDA_PATH", "SHOPPING_AUTONOMOUS_RUN_ID",
                "VLLM_DISABLE_COMPILE_CACHE", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        if environment.get(key):
            overrides.append("+ray_kwargs.ray_init.runtime_env.env_vars." + key + "="
                             + hydra_string(environment[key]))
    if args.profile == "eval":
        if args.method != "grpo" or not args.identity_contexts:
            raise SystemExit("eval requires --method grpo and --identity-contexts; it performs no updates")
        contexts = _validated_path(args.identity_contexts, "identity contexts")
        diagnostic_env = {
            "SHOPPING_DIAGNOSTIC_DIR": str(output / "runtime_audit"),
            "SHOPPING_DIAGNOSTIC_ROLLOUT_DIR": str(output / "task_rollouts"),
            "SHOPPING_DIAGNOSTIC_CONTEXTS": str(contexts),
            "SHOPPING_DIAGNOSTIC_ADAPTER": str(adapter),
        }
        for key in ("CUDA_HOME", "CUDA_PATH", "SHOPPING_AUTONOMOUS_RUN_ID",
                    "VLLM_DISABLE_COMPILE_CACHE", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
            if environment.get(key):
                diagnostic_env[key] = environment[key]
        environment.update(diagnostic_env)
        # Replace existing common overrides rather than passing duplicate keys.
        overrides = [item for item in overrides if not any(item.startswith(
            "+ray_kwargs.ray_init.runtime_env.env_vars." + key + "=")
            for key in diagnostic_env)]
        overrides += [
            "shopping_evaluation.enable=true", "trainer.val_only=true",
            "trainer.val_before_train=true", "trainer.total_training_steps=1",
            "trainer.save_freq=-1", "trainer.test_freq=-1",
            "data.filter_overlong_prompts_workers=4",
            # Keep Qwen3.5 greedy evaluation independent of concurrent batches
            # and avoid copying a small validation batch to eight agent workers.
            "data.val_batch_size=1",
            "actor_rollout_ref.rollout.agent.num_workers=1",
            "actor_rollout_ref.rollout.enable_prefix_caching=false",
            "actor_rollout_ref.rollout.enforce_eager=true",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
            "actor_rollout_ref.rollout.val_kwargs.temperature=0.0",
            "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=false",
            "actor_rollout_ref.actor.calculate_entropy=false",
            "actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=true",
            "trainer.validation_data_dir=" + hydra_string(str(output / "generations")),
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls="
            "shopping_grpo.training.grpo.diagnostic_worker.DiagnosticWorkerExtension",
        ]
        overrides += ["+ray_kwargs.ray_init.runtime_env.env_vars." + key + "=" + hydra_string(value)
                      for key, value in diagnostic_env.items()]
        if evaluation_manifest:
            overrides.append("shopping_evaluation.task_manifest=" + hydra_string(str(evaluation_manifest)))
    if selection:
        overrides += [
            "shopping_checkpoint_selection.enable=true",
            f"shopping_checkpoint_selection.interval={args.validation_interval}",
            f"shopping_checkpoint_selection.patience={args.early_stopping_patience}",
            f"shopping_checkpoint_selection.minimum_updates={args.early_stopping_minimum_updates}",
            f"trainer.test_freq={args.validation_interval}", f"trainer.save_freq={args.validation_interval}",
            "trainer.val_before_train=true", "data.val_batch_size=1", "data.validation_shuffle=false",
            "actor_rollout_ref.rollout.enforce_eager=true", "actor_rollout_ref.rollout.enable_prefix_caching=false",
            "actor_rollout_ref.rollout.val_kwargs.n=1", "actor_rollout_ref.rollout.val_kwargs.temperature=0.0",
            "actor_rollout_ref.rollout.val_kwargs.top_p=1.0", "actor_rollout_ref.rollout.val_kwargs.do_sample=false",
        ]
    if feedback:
        overrides += ["shopping_feedback_sampling.enable=true",
                      "shopping_feedback_sampling.manifest=" + hydra_string(str(feedback)),
                      f"shopping_feedback_sampling.hard_fraction={args.hard_task_fraction}"]
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if any(value.lstrip("+").startswith("shopping_feedback_sampling.") or
           value.lstrip("+").startswith("~shopping_feedback_sampling") for value in extra):
        raise SystemExit("feedback sampling settings must use the dedicated options")
    if selection:
        protected = ("shopping_checkpoint_selection.", "trainer.test_freq=", "trainer.save_freq=",
                     "trainer.val_before_train=", "data.val_batch_size=", "data.validation_shuffle=",
                     "actor_rollout_ref.rollout.enforce_eager=", "actor_rollout_ref.rollout.enable_prefix_caching=",
                     "actor_rollout_ref.rollout.val_kwargs.")
        if any(value.lstrip("+").startswith(protected) for value in extra):
            raise SystemExit("checkpoint selection validation settings must use the dedicated options")
    if flash and any(value.lstrip("+").startswith("shopping_flashreinforce.invalid_batch_policy=")
                     for value in extra):
        raise SystemExit("select the admission policy with --flash-invalid-batch-policy so launch identity stays explicit")
    if flash and any(value.lstrip("+").startswith("shopping_flashreinforce.execution=") for value in extra):
        raise SystemExit("select execution with --flash-execution so launch identity stays explicit")
    if args.profile == "eval":
        # Evaluation is deliberately not an escape hatch for changing the recipe.
        allowed = ("ray_kwargs.ray_init.num_cpus=",)
        if any(not value.startswith(allowed) for value in extra):
            raise SystemExit("eval accepts only the Ray CPU allocation override")
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config.parent}",
        f"--config-name={config.stem}",
        *overrides,
        *extra,
    ]
    return command, environment


def main() -> None:
    args = parse_args()
    command, environment = build_command(args)
    audit = {
        "command": command,
        "model": environment["GRPO_MODEL_PATH"],
        "train_data": environment["GRPO_TRAIN_FILE"],
        "val_data": environment["GRPO_VAL_FILE"],
        "env_url": environment["SHOPSIM_BASE_URL"],
        "output": environment["GRPO_OUTPUT_DIR"],
        "logger": args.logger,
        "config": str(args.config.resolve()),
        "method": args.method,
        "profile": args.profile,
        "training_reward_overlay": args.reward_overlay,
        "parent_adapter_sha256": environment["SHOPPING_PARENT_ADAPTER_SHA256"],
        "gigpo_anchor_schema_version": GIGPO_SCHEMA_VERSION,
        "environment_manifest_sha256": hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest(),
        "environment_runtime_patches": json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8")).get("runtime_patches", {}),
    }
    if args.method == "flashreinforce":
        audit["flashreinforce"] = {
            "prompts_per_batch": args.flash_batch_size,
            "rollouts_per_prompt": 1,
            "optimizer_steps_per_fresh_batch": 1,
            "execution": "asynchronous_single_gpu_colocated" if args.flash_execution == "async" else "synchronous_single_gpu",
            "cross_update_async_queue": args.flash_execution == "async",
            "partial_rollout": args.flash_execution == "async",
            "invalid_batch_policy": args.flash_invalid_batch_policy,
            "admission_protocol": "flash-whole-batch-native-v1" if args.flash_invalid_batch_policy == "skip_batch" else "strict-error",
            "admission_config_from_hydra": True,
        }
    if args.checkpoint_selection:
        audit["checkpoint_selection"] = {
            "protocol": "validation50-serial-eager-native-greedy-v1",
            "interval": args.validation_interval, "patience": args.early_stopping_patience,
            "minimum_updates": args.early_stopping_minimum_updates,
            "primary_metric": "strict_success_count", "ties": "earlier_checkpoint",
            "durable_selection": str(Path(environment["GRPO_OUTPUT_DIR"]) / "checkpoint_selection/best-checkpoint.json"),
        }
    if args.feedback_manifest:
        from shopping_grpo.training.grpo.feedback_sampling import sha256
        audit["feedback_sampling"] = {"manifest_sha256": sha256(args.feedback_manifest),
            "hard_lane_fraction": args.hard_task_fraction, "background": "uniform over full training pool",
            "actual_shares": "audit dispatch, finished, consumed, admitted and skipped separately"}
    if args.evaluation_task_manifest:
        from shopping_grpo.training.grpo.feedback_sampling import sha256
        audit["evaluation_task_manifest_sha256"] = sha256(args.evaluation_task_manifest)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    Path(environment["GRPO_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    (Path(environment["GRPO_OUTPUT_DIR"]) / "launch.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    preflight = [
        sys.executable,
        str(ROOT / "scripts/check_grpo_runtime.py"),
        *command[5:],
    ]
    preflight_status = subprocess.call(preflight, cwd=ROOT, env=environment)
    if preflight_status:
        raise SystemExit(preflight_status)
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
