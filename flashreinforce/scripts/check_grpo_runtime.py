#!/usr/bin/env python3
"""在加载模型前拒绝污染或版本不匹配的 GRPO 环境。"""

from __future__ import annotations

import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path


EXPECTED_VERSIONS = {
    "verl": "0.8.0",
    "vllm": "0.25.1",
    "torch": "2.11.0",
    "transformers": "5.15.0.dev0",
    "ray": "2.56.1",
    "tensordict": "0.10.0",
    "numpy": "2.2.6",
    "swanlab": "0.9.1",
}
EXPECTED_TRANSFORMERS_REVISION = "7ea2320c76117e6742364808a666ef6f2fb40a67"
PATCH_MARKER = "SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V4"
MAX_SAFE_RESPONSE_LENGTH = 20480
MAX_SAFE_SEQUENCE_LENGTH = 24576
CURRENT_RUNTIME_FILES = {
    "observation.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/observation.py",
    "pack_api.py": "environments/ShopSimulator/shop_env/shop_env/pack_api.py",
    "reward.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py",
    "slot_lease_pool.py": "environments/ShopSimulator/shop_env/shop_env/slot_lease_pool.py",
    "web_agent_text_env.py": "environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py",
}


def validate_reward_runtime_files(manifest, root):
    if manifest.get("lease_contract") != "explicit-client-release-v1":
        raise SystemExit(
            "Environment v2.1 manifest must select explicit-client-release-v1"
        )
    expected = manifest.get("runtime_files_sha256")
    files = dict(CURRENT_RUNTIME_FILES)
    patches = manifest.get("runtime_patches", {})
    if patches:
        parser = patches.get("action_parser", {})
        if (set(patches) != {"action_parser"} or parser.get("version") != "shopping-action-parser-v1"
                or parser.get("source_sha256") != "130e0261c5c589ac133f1562f6d718736cda44abe54e2151867f5b17f136a086"
                or parser.get("patched_sha256") != "80f0fb1a3fc1bfa3ff26a3eb8aa5a7978a60f75fa4fd79bb1a7cefdd15e2c2ab"):
            raise SystemExit("unrecognized ShopSimulator runtime patch")
        files["engine.py"] = "environments/ShopSimulator/shop_env/web_agent_site/engine/engine.py"
        if expected is None or expected.get("engine.py") != parser["patched_sha256"]:
            raise SystemExit("action parser patch must be pinned in the runtime file manifest")
    if not isinstance(expected, dict) or set(expected) != set(files):
        raise SystemExit(
            "Environment v2.1 manifest runtime_files_sha256 is missing or incomplete"
        )
    from shopping_grpo.environment.manifest import sha256_file

    mismatches = {}
    for name, relative_path in files.items():
        actual = sha256_file(Path(root) / relative_path)
        if actual != expected[name]:
            mismatches[name] = {"expected": expected[name], "actual": actual}
    if mismatches:
        raise SystemExit(
            "Environment v2.1 runtime file hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_environment_contract():
    required_version = os.environ.get(
        "SHOPPING_ENVIRONMENT_VERSION",
        "shopsimulator-environment-v2.1",
    )
    if required_version != "shopsimulator-environment-v2.1":
        raise SystemExit(
            "this repository supports only shopsimulator-environment-v2.1"
        )
    manifest_path = os.environ.get("SHOPPING_ENV_MANIFEST")
    if not manifest_path or not Path(manifest_path).is_file():
        raise SystemExit(
            f"{required_version} requires SHOPPING_ENV_MANIFEST pointing to a frozen manifest"
        )
    try:
        from shopping_grpo.environment.manifest import validate_manifest

        manifest = validate_manifest(
            json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        )
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {required_version} manifest: {exc}") from exc
    actual_environment_version = manifest.get(
        "environment_version",
        "shopsimulator-environment-v2.1",
    )
    if actual_environment_version != required_version:
        raise SystemExit(
            "environment manifest version mismatch: "
            f"expected {required_version}, got {actual_environment_version}"
        )
    tools_path = Path(
        os.environ.get(
            "SHOPPING_TOOL_CONFIG",
            Path(__file__).resolve().parents[1]
            / "configs/tools.json",
        )
    )
    tools = json.loads(tools_path.read_text(encoding="utf-8")).get("tools", [])
    tool_names = {
        item.get("tool_schema", {}).get("function", {}).get("name")
        for item in tools
    }
    if "finish_without_purchase" not in tool_names:
        raise SystemExit("Environment v2 tool config is missing finish_without_purchase")
    if int(manifest["max_steps"]) != 35:
        raise SystemExit("Environment v2 GRPO contract requires max_steps=35")
    validate_reward_runtime_files(
        manifest,
        Path(__file__).resolve().parents[1],
    )
    print(
        f"{required_version} manifest preflight passed: "
        + json.dumps(
            {
                "manifest": str(Path(manifest_path).resolve()),
                "shopsimulator_commit": manifest["shopsimulator_commit"],
                "observation_version": manifest["observation_version"],
                "reward_version": manifest["reward"]["version"],
                "search_version": manifest["search"]["version"],
                "lease_contract": manifest.get("lease_contract"),
                "runtime_file_count": len(manifest.get("runtime_files_sha256") or {}),
            },
            sort_keys=True,
        )
    )


def compose_runtime_config(overrides):
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
    except ImportError as exc:
        raise SystemExit(f"cannot parse GRPO config before preflight: {exc}") from exc

    GlobalHydra.instance().clear()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    config_name = os.environ.get("GRPO_CONFIG_NAME", "grpo")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=list(overrides))


def validate_transformers_revision():
    """The Qwen3.5 runtime uses one pinned upstream Transformers revision."""
    dist = distribution("transformers")
    direct_url = Path(dist.locate_file("transformers-5.15.0.dev0.dist-info/direct_url.json"))
    if not direct_url.is_file():
        raise SystemExit(
            "cannot verify pinned Transformers revision: direct_url.json is missing"
        )
    try:
        metadata = json.loads(direct_url.read_text(encoding="utf-8"))
        revision = metadata.get("vcs_info", {}).get("commit_id")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid Transformers direct_url.json: {exc}") from exc
    if revision is None and metadata.get("url", "").startswith("file:"):
        from shopping_grpo.training.grpo.dependency_provenance import verify_python_source
        source_manifest = Path(__file__).resolve().parents[1] / "data/runtime/transformers-source.json"
        try:
            count = verify_python_source(
                json.loads(source_manifest.read_text(encoding="utf-8")),
                Path(dist.locate_file("transformers")), EXPECTED_TRANSFORMERS_REVISION,
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot verify local Transformers source install: {exc}") from exc
        print(f"pinned Transformers source preflight passed: {count} exact Python file hashes")
        return
    if revision != EXPECTED_TRANSFORMERS_REVISION:
        raise SystemExit(
            "incompatible Transformers revision: expected "
            f"{EXPECTED_TRANSFORMERS_REVISION}, got {revision}"
        )
    print(f"pinned Transformers revision preflight passed: {revision}")


def validate_dynamic_sampling(config, verl_source: Path, installed):
    dynamic_config = config.get("shopping_dynamic_sampling", {})
    flash_enabled = config.get("shopping_flashreinforce", {}).get("enable", False)
    if not bool(dynamic_config.get("enable", False)) and not flash_enabled:
        return

    if installed.get("verl") != "0.8.0":
        raise SystemExit(
            f"shopping dynamic sampling requires verl==0.8.0, got {installed.get('verl')}"
        )
    ray_trainer = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    if not ray_trainer.is_file():
        raise SystemExit(f"cannot locate installed RayPPOTrainer source: {ray_trainer}")
    if PATCH_MARKER not in ray_trainer.read_text(encoding="utf-8"):
        raise SystemExit(
            "shopping dynamic sampling is enabled but the pinned veRL patch marker is missing; "
            "run scripts/apply_verl_dynamic_sampling_patch.py first"
        )

    if flash_enabled:
        # V4 also provides post-unpad validation persistence and optimizer-step
        # diagnostics used by this arm; its group-filter branch stays disabled.
        from shopping_grpo.training.grpo.flashreinforce import validate_flashreinforce_config
        validate_flashreinforce_config(config)
        if config.shopping_flashreinforce.get("invalid_batch_policy", "error") == "skip_batch":
            from shopping_grpo.training.grpo.flash_batching import prepare_fit
            prepare_fit(ray_trainer.read_text(encoding="utf-8"), {},
                        asynchronous=config.shopping_flashreinforce.get("execution", "sync") == "async")
            if config.shopping_flashreinforce.get("execution", "sync") == "async":
                import hashlib
                pin = Path(__file__).resolve().parents[1] / "patches/verl-0.8.0-flash-async-sources.json"
                for name, digest in json.loads(pin.read_text(encoding="utf-8")).items():
                    path = verl_source.parent / name
                    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                        raise SystemExit(f"async veRL runtime source changed: {name}")
                print("FlashREINFORCE async client/server/checkpoint source pins passed")
            if not os.environ.get("SHOPPING_GRPO_DIAGNOSTICS_PATH"):
                raise SystemExit("FlashREINFORCE whole-batch admission requires diagnostics persistence")
            print("FlashREINFORCE whole-batch admission hook preflight passed")
        print("FlashREINFORCE V4 trainer diagnostics preflight passed (group filtering disabled)")
        return

    try:
        from shopping_grpo.training.grpo.dynamic_sampling import (
            extract_shopping_group_signals,
            select_reward_varying_groups,
        )
    except ImportError as exc:
        raise SystemExit(f"shopping dynamic sampling helper is unavailable: {exc}") from exc
    utility, success, invalid, reasons = extract_shopping_group_signals(
        [
            {
                "infrastructure_invalid": False,
                "reward": {
                    "terminal_utility": reward,
                    "purchase_success": reward > 0,
                    "sampling_invalid": False,
                },
            }
            for reward in (0.0, 1.0, 0.0, 0.0)
        ]
    )
    indices, _ = select_reward_varying_groups(
        ["preflight"] * 4,
        [0.0, 1.0, 0.0, 0.0],
        terminal_utilities=utility,
        purchase_success=success,
        sampling_invalid=invalid,
        sampling_invalid_reasons=reasons,
    )
    if indices != [0, 1, 2, 3]:
        raise SystemExit("shopping dynamic sampling helper failed its import-time sanity check")

    if dynamic_config.get("metric") != "seq_reward":
        raise SystemExit("shopping_dynamic_sampling.metric must be seq_reward")
    if int(dynamic_config.get("max_num_gen_batches", 0)) <= 0:
        raise SystemExit("shopping_dynamic_sampling.max_num_gen_batches must be positive")
    if int(dynamic_config.get("max_consecutive_skipped_updates", 0)) <= 0:
        raise SystemExit(
            "shopping_dynamic_sampling.max_consecutive_skipped_updates must be positive"
        )
    reward_tolerance = float(dynamic_config.get("reward_tolerance", -1))
    if reward_tolerance < 0 or not math.isfinite(reward_tolerance):
        raise SystemExit("shopping_dynamic_sampling.reward_tolerance must be finite and non-negative")
    if not bool(config.algorithm.rollout_correction.get("bypass_mode", False)):
        raise SystemExit("shopping dynamic sampling requires rollout_correction.bypass_mode=true")
    if not bool(config.actor_rollout_ref.rollout.get("calculate_log_probs", False)):
        raise SystemExit("shopping dynamic sampling requires rollout.calculate_log_probs=true")

    print(
        "shopping dynamic sampling preflight passed: "
        + json.dumps(
            {
                "enable": True,
                "metric": str(dynamic_config.metric),
                "max_num_gen_batches": int(dynamic_config.max_num_gen_batches),
                "max_consecutive_skipped_updates": int(
                    dynamic_config.max_consecutive_skipped_updates
                ),
                "reward_tolerance": reward_tolerance,
                "ray_trainer": str(ray_trainer),
                "marker": PATCH_MARKER,
            },
            sort_keys=True,
        )
    )


def validate_trace(config):
    """Fail before model loading when the optional TRACE arm is inconsistent."""
    trace = config.get("shopping_trace", {})
    if not bool(trace.get("enable", False)):
        return
    if str(config["algorithm"]["adv_estimator"]).lower() != "grpo":
        raise SystemExit("shopping TRACE only supports GRPO")
    model = config["actor_rollout_ref"]["model"]
    lora_rank = int(model.get("lora", {}).get("rank", 0) or model.get("lora_rank", 0))
    if lora_rank <= 0:
        raise SystemExit("shopping TRACE requires LoRA to expose a frozen base reference")
    if int(config["trainer"]["n_gpus_per_node"]) != 1 or int(config["trainer"]["nnodes"]) != 1:
        raise SystemExit("shopping TRACE currently supports the repository's single-GPU recipe")
    values = {
        "epsilon": float(trace.get("epsilon", 0)),
        "discount": float(trace.get("discount", -1)),
        "terminal_weight": float(trace.get("terminal_weight", -1)),
        "outcome_weight": float(trace.get("outcome_weight", -1)),
        "turn_weight": float(trace.get("turn_weight", -1)),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise SystemExit("shopping TRACE hyperparameters must be finite")
    if values["epsilon"] <= 0 or int(trace.get("horizon", 0)) <= 0:
        raise SystemExit("shopping TRACE epsilon and horizon must be positive")
    if not 0 <= values["discount"] <= 1:
        raise SystemExit("shopping TRACE discount must be in [0, 1]")
    if min(values["terminal_weight"], values["outcome_weight"], values["turn_weight"]) < 0:
        raise SystemExit("shopping TRACE weights must be non-negative")
    if int(trace.get("max_sequence_length", 0)) != MAX_SAFE_SEQUENCE_LENGTH:
        raise SystemExit(
            f"shopping TRACE max_sequence_length must equal {MAX_SAFE_SEQUENCE_LENGTH}"
        )


def validate_swanlab_tracking(config):
    """Validate SwanLab only when the user explicitly enables it."""
    logger_backends = list(config.trainer.get("logger", []))
    if "swanlab" not in logger_backends:
        return
    forbidden = {"wandb", "tracking", "vemlp_wandb"} & set(logger_backends)
    if forbidden:
        raise SystemExit(
            "Reward v3 GRPO forbids W&B logger backends: "
            + ", ".join(sorted(forbidden))
        )
    if os.environ.get("SWANLAB_MODE") != "online":
        raise SystemExit("Reward v3 GRPO requires SWANLAB_MODE=online")
    if not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit(
            "Reward v3 GRPO requires SWANLAB_API_KEY in the launching environment"
        )
    log_dir = os.environ.get("SWANLAB_LOG_DIR")
    if not log_dir:
        raise SystemExit("Reward v3 GRPO requires SWANLAB_LOG_DIR")
    resolved_log_dir = Path(log_dir).resolve()
    if str(config.trainer.get("project_name")) != "shopping-grpo":
        raise SystemExit("Reward v3 GRPO SwanLab project must be shopping-grpo")
    print(
        "SwanLab online preflight passed: "
        + json.dumps(
            {
                "api_key": "present",
                "logger": logger_backends,
                "log_dir": str(resolved_log_dir),
                "mode": "online",
                "project": str(config.trainer.project_name),
                "run_name": str(config.trainer.experiment_name),
            },
            sort_keys=True,
        )
    )


def ppo_gradient_accumulation_steps(mini_batch_size: int, micro_batch_size: int) -> int:
    mini = int(mini_batch_size)
    micro = int(micro_batch_size)
    if mini <= 0 or micro <= 0:
        raise ValueError("PPO mini and micro batch sizes must be positive")
    if mini % micro:
        raise ValueError("PPO mini batch size must be divisible by micro batch size")
    return mini // micro


def validate_training_memory_budget(config):
    prompt_length = int(config.data.max_prompt_length)
    response_length = int(config.data.max_response_length)
    total_length = prompt_length + response_length
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    reference = config.actor_rollout_ref.ref

    if response_length > MAX_SAFE_RESPONSE_LENGTH:
        raise SystemExit(
            "unsafe GRPO response budget: "
            f"max_response_length={response_length} exceeds {MAX_SAFE_RESPONSE_LENGTH}"
        )
    if total_length > MAX_SAFE_SEQUENCE_LENGTH:
        raise SystemExit(
            "unsafe GRPO sequence budget: "
            f"max_prompt_length + max_response_length = {total_length}, "
            f"limit is {MAX_SAFE_SEQUENCE_LENGTH}"
        )
    for name, value in (
        ("rollout.max_model_len", int(rollout.max_model_len)),
        ("rollout.max_num_batched_tokens", int(rollout.max_num_batched_tokens)),
        (
            "rollout.log_prob_max_token_len_per_gpu",
            int(rollout.log_prob_max_token_len_per_gpu),
        ),
        ("actor.ppo_max_token_len_per_gpu", int(actor.ppo_max_token_len_per_gpu)),
        ("ref.log_prob_max_token_len_per_gpu", int(reference.log_prob_max_token_len_per_gpu)),
    ):
        if value != MAX_SAFE_SEQUENCE_LENGTH:
            raise SystemExit(
                f"unsafe or inconsistent GRPO memory budget: {name} must equal "
                f"{MAX_SAFE_SEQUENCE_LENGTH}, got {value}"
            )
    if bool(actor.use_dynamic_bsz):
        raise SystemExit(
            "actor.use_dynamic_bsz must be false so configured PPO micro batches are enforced"
        )
    actor_micro_batch_size = int(actor.ppo_micro_batch_size_per_gpu)
    actor_mini_batch_size = int(actor.ppo_mini_batch_size)
    # veRL 0.8 multiplies prompt mini-batch size by rollout.n before actor update.
    actor_mini_trajectories = actor_mini_batch_size * int(rollout.n)
    try:
        gradient_accumulation_steps = ppo_gradient_accumulation_steps(
            actor_mini_trajectories,
            actor_micro_batch_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if bool(rollout.log_prob_use_dynamic_bsz):
        raise SystemExit(
            "rollout.log_prob_use_dynamic_bsz must be false so "
            "log_prob_micro_batch_size_per_gpu=1 is enforced"
        )
    if int(rollout.log_prob_micro_batch_size_per_gpu) != 1:
        raise SystemExit("rollout.log_prob_micro_batch_size_per_gpu must equal 1")
    if bool(reference.log_prob_use_dynamic_bsz):
        raise SystemExit(
            "ref.log_prob_use_dynamic_bsz must be false so "
            "log_prob_micro_batch_size_per_gpu=1 is enforced"
        )
    if int(reference.log_prob_micro_batch_size_per_gpu) != 1:
        raise SystemExit("ref.log_prob_micro_batch_size_per_gpu must equal 1")

    print(
        "GRPO training memory budget preflight passed: "
        + json.dumps(
            {
                "max_prompt_length": prompt_length,
                "max_response_length": response_length,
                "max_sequence_length": total_length,
                "actor_mini_batch_size": actor_mini_batch_size,
                "actor_mini_trajectories": actor_mini_trajectories,
                "actor_micro_batch_size_per_gpu": actor_micro_batch_size,
                "actor_gradient_accumulation_steps": gradient_accumulation_steps,
                "actor_dynamic_batch": False,
                "rollout_log_prob_micro_batch_size_per_gpu": 1,
                "rollout_log_prob_dynamic_batch": False,
                "reference_micro_batch_size_per_gpu": 1,
                "reference_dynamic_batch": False,
            },
            sort_keys=True,
        )
    )


def validate_runtime_chat_templates(driver_tokenizer, processor, actor_tokenizer):
    """veRL TaskRunner uses model.path even when the actor overrides tokenizer_path."""
    expected = actor_tokenizer.chat_template
    if not expected or driver_tokenizer.chat_template != expected:
        raise SystemExit("driver tokenizer must retain the parent SFT chat template at model.path")
    if processor is not None and processor.chat_template != expected:
        raise SystemExit("model.path processor must retain the parent SFT chat template")


def validate_model_prompt_contract(config):
    from verl.utils import hf_processor, hf_tokenizer

    model_path = str(config.actor_rollout_ref.model.path)
    actor_path = str(config.actor_rollout_ref.model.get("tokenizer_path") or model_path)
    trust = bool(config.data.get("trust_remote_code", False))
    driver = hf_tokenizer(model_path, trust_remote_code=trust)
    processor = hf_processor(model_path, trust_remote_code=trust, use_fast=True)
    actor = hf_tokenizer(actor_path, trust_remote_code=trust)
    validate_runtime_chat_templates(driver, processor, actor)
    print("driver/actor/processor SFT chat template preflight passed")


def main():
    cpu_only = "--cpu-only" in sys.argv[1:]
    overrides = [value for value in sys.argv[1:] if value != "--cpu-only"]
    config = compose_runtime_config(overrides)
    validate_environment_contract()
    required_paths = {
        "GRPO_TRAIN_FILE": os.environ.get("GRPO_TRAIN_FILE"),
        "GRPO_VAL_FILE": os.environ.get("GRPO_VAL_FILE"),
    }
    missing = [name for name, value in required_paths.items() if not value or not Path(value).is_file()]
    if missing:
        raise SystemExit("missing GRPO parquet file(s): " + ", ".join(missing))
    validate_training_task_isolation(config)
    validate_training_memory_budget(config)
    validate_trace(config)
    from shopping_grpo.training.grpo.gigpo import validate_gigpo_config
    validate_gigpo_config(config)
    from shopping_grpo.training.grpo.flashreinforce import validate_flashreinforce_config
    validate_flashreinforce_config(config)
    from shopping_grpo.training.grpo.reward_overlay import MODES
    overlay = os.environ.get("SHOPPING_TRAINING_REWARD_OVERLAY", "native")
    if overlay not in MODES:
        raise SystemExit("unsupported training reward overlay")
    if config.get("shopping_evaluation", {}).get("enable", False) and overlay != "native":
        raise SystemExit("diagnostic evaluation requires the native reward")
    if overlay != "native" and config.get("shopping_gigpo", {}).get("enable", False):
        raise SystemExit("the first reward comparison must use GRPO without GiGPO")
    if overlay != "native" and config.get("shopping_flashreinforce", {}).get("enable", False):
        raise SystemExit("the first FlashREINFORCE comparison requires native Reward v3")

    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"incompatible Python: expected 3.12, got {sys.version.split()[0]}")

    installed = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            installed[package] = version(package)
        except PackageNotFoundError as exc:
            raise SystemExit(f"missing GRPO dependency: {package}=={expected}") from exc
        if installed[package].split("+", 1)[0] != expected:
            raise SystemExit(
                f"incompatible GRPO dependency: expected {package}=={expected}, got {installed[package]}"
            )
    validate_transformers_revision()
    validate_model_prompt_contract(config)

    try:
        import torch
        import verl
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
        from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
        from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
        from shopping_grpo.training.grpo.compat import install_torch_padding_fallback
        from verl.tools.base_tool import BaseTool
        from verl.utils.tracking import Tracking
    except ImportError as exc:
        raise SystemExit(
            "incompatible veRL 0.8 install: required AgentLoop/Tool APIs are unavailable; "
            f"original error: {exc}"
        ) from exc

    verl_source = Path(verl.__file__).resolve()
    if not cpu_only and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the GRPO environment")
    if (
        not issubclass(ShoppingToolAgentLoop, ToolAgentLoop)
        or not issubclass(ShopSimulatorTool, BaseTool)
        or AgentState.TERMINATED.value != "terminated"
        or not hasattr(ToolAgentLoop, "_handle_processing_tools_state")
    ):
        raise SystemExit("incompatible veRL ToolAgentLoop lifecycle API")
    if "qwen3_coder" not in ToolParser._registry:
        raise SystemExit("veRL 0.8 built-in qwen3_coder parser is unavailable")
    if "swanlab" not in Tracking.supported_backend:
        raise SystemExit("veRL 0.8 SwanLab tracking backend is unavailable")
    validate_dynamic_sampling(config, verl_source, installed)
    validate_swanlab_tracking(config)
    install_torch_padding_fallback()
    print(
        ("GRPO CPU preparation preflight passed (CUDA checks deferred): " if cpu_only
         else "GRPO runtime preflight passed: ")
        + ", ".join(f"{name}={value}" for name, value in installed.items())
        + f", source={verl_source}"
    )


def validate_training_task_isolation(config):
    """Reject leakage before model loading; a probe intentionally uses train IDs."""
    import pyarrow.parquet as pq
    root = Path(__file__).resolve().parents[1]
    final = {int(json.loads(line)["task_id"]) for line in
             (root / "data/evaluation/tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}

    def ids(paths):
        if isinstance(paths, str):
            paths = [paths]
        result = set()
        for path in paths:
            for row in pq.read_table(path, columns=["extra_info"]).to_pylist():
                result.add(int(row["extra_info"]["task_id"]))
        return result

    train, validation = ids(config.data.train_files), ids(config.data.val_files)
    if not train or not validation or (train | validation) & final:
        raise SystemExit("GRPO train/validation contains Final tasks or is empty")
    if config.get("shopping_feedback_sampling", {}).get("enable", False):
        from shopping_grpo.training.grpo.feedback_sampling import load_manifest, validate_feedback_config
        validate_feedback_config(config)
        load_manifest(config.shopping_feedback_sampling.manifest, config.data.train_files, root=root)
    if config.get("shopping_checkpoint_selection", {}).get("enable", False):
        from shopping_grpo.training.grpo.checkpoint_selection import validate_selection_config
        validate_selection_config(config)
        frozen = {int(json.loads(line)["task_id"]) for line in
                  (root / "data/grpo/validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
        if validation != frozen or len(validation) != 50 or train & validation:
            raise SystemExit("checkpoint selection requires the unchanged isolated validation50")
    if config.get("shopping_evaluation", {}).get("enable", False):
        if not config.trainer.get("val_only", False) or not config.trainer.val_before_train:
            raise SystemExit("diagnostic evaluation must be validation-only")
        frozen = {int(json.loads(line)["task_id"]) for line in
                  (root / "data/grpo/validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
        task_manifest = config.shopping_evaluation.get("task_manifest")
        if task_manifest:
            from shopping_grpo.evaluation.frozen_tasks import validate_frozen_evaluation
            paths = config.data.val_files
            paths = [paths] if isinstance(paths, str) else list(paths)
            if len(paths) != 1:
                raise SystemExit("confirmation evaluation requires one frozen parquet")
            manifest = validate_frozen_evaluation(task_manifest, paths[0], root=root)
            frozen = set(manifest["task_ids"])
        if train & validation or validation != frozen or len(validation) != (200 if task_manifest else 50):
            raise SystemExit("diagnostic evaluation must use unchanged isolated frozen tasks")
        if config.actor_rollout_ref.rollout.val_kwargs.n != 1:
            raise SystemExit("diagnostic evaluation requires G1")
    elif config.trainer.get("val_only", False):
        if not validation <= train:
            raise SystemExit("no-gradient probe must use a frozen subset of training tasks")
    elif train & validation:
        raise SystemExit("GRPO training and validation task IDs overlap")
    print(f"GRPO task isolation passed: train={len(train)}, validation/probe={len(validation)}")


if __name__ == "__main__":
    main()
