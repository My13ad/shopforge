"""Opt-in, read-only receipts for native validation and weight synchronization."""
from __future__ import annotations

from contextlib import nullcontext
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
from uuid import uuid4

_validation_capture = ContextVar("shopping_checkpoint_validation", default=None)


@contextmanager
def capture_validation(step):
    record = {"step": step, "infos": []}
    token = _validation_capture.set(record)
    try:
        yield record["infos"]
    finally:
        _validation_capture.reset(token)


def write_receipt(directory, kind, payload):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "kind": kind,
              "adapter_sha256": os.environ.get("SHOPPING_PARENT_ADAPTER_SHA256", ""),
              **payload}
    path = root / f"{kind}-{uuid4().hex}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def strict_success(info):
    return (info.get("done") is True and info.get("terminal_done") is True
            and info.get("terminal_over") is True
            and info.get("reward_type") == "gold_purchase"
            and info.get("reward_valid") is True
            and not info.get("infrastructure_invalid")
            and not info.get("reward_unverifiable")
            and not info.get("reward", {}).get("sampling_invalid", False))


def annotate_rollout_statistics(info, output):
    """Attach unpadded token counts in memory; workers never write task receipts."""
    mask = list(output.response_mask)
    if len(mask) != len(output.response_ids) or any(item not in (0, 1) for item in mask):
        raise ValueError("invalid response mask in evaluation receipt")
    info = dict(info)
    info.update(strict_success=strict_success(info), policy_reward=float(output.reward_score),
                assistant_tokens=sum(mask), response_tokens=len(mask),
                tool_and_template_tokens=len(mask) - sum(mask),
                prompt_tokens=len(output.prompt_ids), native_num_turns=output.num_turns)
    return info


def persist_unpadded_validation_batch(batch, *, global_step):
    """Called only after the pinned trainer's native unpad; no deduplication."""
    directory = os.environ.get("SHOPPING_DIAGNOSTIC_ROLLOUT_DIR")
    validation_directory = os.environ.get("SHOPPING_VALIDATION_RECEIPTS_DIR")
    probe_directory = os.environ.get("SHOPPING_GIGPO_PROBE_DIR")
    capture = _validation_capture.get()
    if not (directory or validation_directory or probe_directory) and capture is None:
        return
    infos = batch.non_tensor_batch.get("shopping")
    if infos is None or len(infos) != len(batch):
        raise ValueError("unpad receipt hook requires one shopping record per native row")
    prepared = []
    for info in infos:
        if not isinstance(info, dict) or "assistant_tokens" not in info:
            raise ValueError("missing AgentLoop statistics after native unpad")
        if not math.isfinite(float(info["policy_reward"])):
            raise ValueError("nonfinite validation policy reward")
        prepared.append(info)
    if capture is not None:
        if int(global_step) != capture["step"]:
            raise ValueError("validation receipts do not match the committed checkpoint")
        capture["infos"].extend(deepcopy(prepared))
    for info in prepared:
        payload = {"shopping": info, "receipt_scope": "post_native_unpad",
                   "global_step": int(global_step)}
        if directory:
            write_receipt(directory, "task", payload)
        if validation_directory:
            write_receipt(Path(validation_directory) / f"step-{global_step}", "task", payload)
        if probe_directory:
            from shopping_grpo.training.grpo.probe import persist_probe_rollout
            persist_probe_rollout(info)


async def run_identity_audit(loop, state, agent_data):
    directory = os.environ.get("SHOPPING_DIAGNOSTIC_DIR")
    if not directory or state is None or state.get("_diagnostic_audited"):
        return
    contexts = json.loads(Path(os.environ["SHOPPING_DIAGNOSTIC_CONTEXTS"]).read_text(encoding="utf-8"))
    if int(state["task_id"]) != int(contexts["trigger_validation_task_id"]):
        return
    state["_diagnostic_audited"] = True
    # Independent worker processes can see padding copies of the trigger task.
    # One atomic directory claim permits one identity audit per fresh arm/run.
    claim = Path(directory) / "identity-claim"
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir()
    except FileExistsError:
        return
    for context in contexts["contexts"]:
        prompt = await loop.apply_chat_template(
            context["messages"], tools=getattr(agent_data, "_active_tool_schemas", loop.tool_schemas))
        if len(prompt) > 4096:
            raise ValueError("identity context exceeds the frozen prompt budget")
        repeats = []
        for repeat in range(2):
            result = await loop.server_manager.generate(
                request_id=f"identity-{uuid4().hex}", prompt_ids=prompt,
                sampling_params={"max_tokens": 1, "temperature": 0.0, "top_p": 1.0,
                                 "logprobs": True, "prompt_logprobs": 0})
            conditional = result.extra_fields.get("prompt_logprobs")
            if not conditional or not result.log_probs:
                raise RuntimeError("inference identity request did not return conditional logprobs")
            repeats.append({"repeat": repeat, "token_ids": result.token_ids,
                            "log_probs": result.log_probs, "prompt_logprobs": conditional,
                            "extra_prompt_ids": result.extra_fields.get("prompt_ids"),
                            "global_steps": result.extra_fields.get("global_steps")})
        write_receipt(directory, "identity", {
            "training_task_id": context["task_id"], "prompt_tokens": len(prompt),
            "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
            "repeats": repeats})


def normalize_adapter_name(name):
    return name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "").replace(".default.", ".")


def verify_actor_adapter(module, adapter_directory):
    """Compare every live LoRA parameter against the selected file, in live dtype."""
    import torch
    from safetensors.torch import load_file
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    context = FSDP.summon_full_params(module, recurse=True, writeback=False) if isinstance(module, FSDP) else nullcontext()
    with context:
        return _verify_unflattened_adapter(module, adapter_directory, load_file, torch)


def _verify_unflattened_adapter(module, adapter_directory, load_file, torch):
    expected = {normalize_adapter_name(k): v for k, v in
                load_file(str(Path(adapter_directory) / "adapter_model.safetensors"), device="cpu").items()}
    observed = {}
    dtypes = set()
    for name, param in module.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        key = normalize_adapter_name(name)
        if key not in expected:
            raise RuntimeError(f"unidentified live adapter tensor: {key}")
        value = param.detach()
        if hasattr(value, "to_local"):
            local = value.to_local()
            if tuple(local.shape) != tuple(value.shape):
                raise RuntimeError("diagnostic adapter audit requires an unsharded single GPU")
            value = local
        value = value.cpu()
        dtypes.add(str(value.dtype))
        if not torch.equal(value, expected[key].to(value.dtype)):
            raise RuntimeError(f"live adapter differs from the selected checkpoint: {key}")
        observed[key] = True
    if set(observed) != set(expected) or not observed:
        raise RuntimeError("incomplete live adapter identity audit")
    return {"all_tensors_match_selected_file_in_live_dtype": True,
            "tensor_count": len(observed), "dtypes": sorted(dtypes)}


def weight_sample(name, tensor):
    # Down projections are unfused in this model; identical suffixes can be read
    # back from the receiving vLLM model without assuming a qkv packing layout.
    if ".layers." not in name or not name.endswith(".mlp.down_proj.weight"):
        return None
    import torch
    if hasattr(tensor, "full_tensor"):
        tensor = tensor.full_tensor()
    flat = tensor.detach().reshape(-1)
    indices = torch.linspace(0, flat.numel() - 1, min(64, flat.numel()), device=flat.device).long()
    values = flat[indices].to(torch.bfloat16).float().cpu().tolist()
    return {"name_suffix": name[name.index("layers."):], "shape": list(tensor.shape),
            "source_dtype": str(tensor.dtype), "values_bf16": values}


def samples_equal(left, right):
    return (left["name_suffix"] == right["name_suffix"] and left["shape"] == right["shape"]
            and left["values_bf16"] == right["values_bf16"])


def install_runtime_audit_hooks():
    if not os.environ.get("SHOPPING_DIAGNOSTIC_DIR"):
        return
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter
    if not getattr(FSDPEngine.get_per_tensor_param, "_shopping_diagnostic", False):
        original = FSDPEngine.get_per_tensor_param

        @wraps(original)
        def audited_params(self, *args, **kwargs):
            if not getattr(self, "_shopping_adapter_verified", False):
                # Match the native method's first operation before summoning:
                # these FSDP1 parameters have been offloaded to CPU, while their
                # compute device remains CUDA. The original load below is then
                # a no-op, and the original sync still owns normal offloading.
                from verl.utils.fsdp_utils import load_fsdp_model_to_gpu
                if not self._uses_fsdp2_cpu_offload_policy:
                    load_fsdp_model_to_gpu(self.module)
                receipt = verify_actor_adapter(self.module, os.environ["SHOPPING_DIAGNOSTIC_ADAPTER"])
                write_receipt(os.environ["SHOPPING_DIAGNOSTIC_DIR"], "actor_adapter", receipt)
                self._shopping_adapter_verified = True
            return original(self, *args, **kwargs)
        audited_params._shopping_diagnostic = True
        FSDPEngine.get_per_tensor_param = audited_params
    if not getattr(ServerAdapter.update_weights, "_shopping_diagnostic", False):
        original_update = ServerAdapter.update_weights

        @wraps(original_update)
        async def audited_update(self, weights, *args, **kwargs):
            samples = []

            def tapped():
                for name, value in weights:
                    sample = weight_sample(name, value)
                    if sample is not None:
                        samples.append(sample)
                    yield name, value
            result = await original_update(self, tapped(), *args, **kwargs)
            if not samples:
                raise RuntimeError("no merged down-projection weights observed in synchronization")
            write_receipt(os.environ["SHOPPING_DIAGNOSTIC_DIR"], "sender",
                          {"samples": samples, "completed": True, "global_steps": kwargs.get("global_steps")})
            return result
        audited_update._shopping_diagnostic = True
        ServerAdapter.update_weights = audited_update


def validate_task_records(records, expected_tasks, adapter_sha):
    by_task = {}
    for record in records:
        if record.get("receipt_scope") != "post_native_unpad":
            raise ValueError("evaluation receipts must come after native unpadding")
        if not adapter_sha or record.get("adapter_sha256") != adapter_sha:
            raise ValueError("unidentified or different adapter in evaluation")
        info = record["shopping"]
        task = int(info["task_id"])
        if task not in expected_tasks or task in by_task:
            raise ValueError("unexpected or duplicate validation task")
        if not math.isfinite(float(info["policy_reward"])):
            raise ValueError("nonfinite policy reward")
        if info["strict_success"] != strict_success(info):
            raise ValueError("strict-success receipt is inconsistent")
        by_task[task] = info
    if set(by_task) != expected_tasks:
        raise ValueError("missing validation tasks; denominator cannot be reduced")
    return by_task


def compare_tasks(arms):
    keys = list(arms)
    tasks = set(arms[keys[0]])
    if not tasks or any(set(arm) != tasks for arm in arms.values()):
        raise ValueError("comparison requires the same nonempty task set")
    common = {t for t in tasks if all(strict_success(arm[t]) for arm in arms.values())}
    summary = {}
    for name, arm in arms.items():
        success = {t for t in tasks if strict_success(arm[t])}
        mean = lambda field, ids: sum(float(arm[t][field]) for t in ids) / len(ids) if ids else None
        summary[name] = {"tasks": len(tasks), "strict_successes": len(success),
                         "strict_success_rate": len(success) / len(tasks),
                         "mean_reward": mean("policy_reward", tasks),
                         "infrastructure_invalid": sum(bool(i.get("infrastructure_invalid")) for i in arm.values()),
                         "mean_steps": mean("steps", tasks), "mean_assistant_tokens": mean("assistant_tokens", tasks),
                         "common_success_mean_steps": mean("steps", common),
                         "common_success_mean_assistant_tokens": mean("assistant_tokens", common)}
    pairs = {}
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            pairs[f"{left}_to_{right}"] = {
                "gains": sorted(t for t in tasks if not strict_success(arms[left][t]) and strict_success(arms[right][t])),
                "regressions": sorted(t for t in tasks if strict_success(arms[left][t]) and not strict_success(arms[right][t])),
                "different_actions": sorted(t for t in tasks if arms[left][t]["actions"] != arms[right][t]["actions"])}
    return {"arms": summary, "pairs": pairs, "common_success_tasks": sorted(common)}
