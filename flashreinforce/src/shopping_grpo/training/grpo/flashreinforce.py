"""FlashREINFORCE, PDF Eqs. (5)--(10) and Appendix A's numerical guard.

The optimizer supports synchronous and cross-update asynchronous single-GPU
execution. Completion scheduling and partial rollout live in the async modules.
"""
from __future__ import annotations

import math
from functools import wraps

import torch

METHOD = "flashreinforce"
LOG_RATIO_LIMIT = 30.0
PROBABILITY_EPSILON = 1e-6  # Same Bernoulli projection guard as the Molt reference.


def _mask(response_mask: torch.Tensor) -> torch.Tensor:
    if response_mask.ndim != 2 or not response_mask.numel():
        raise ValueError("FlashREINFORCE needs a nonempty 2D response_mask")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must be binary")
    mask = response_mask.bool()
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("every trajectory must contain policy tokens")
    return mask


def compute_flashreinforce_advantage(token_level_rewards, response_mask, index=None,
                                    config=None, **kwargs):
    """Center the complete batch once, before microbatching or sequence rejection."""
    mask = _mask(response_mask)
    if token_level_rewards.shape != mask.shape or not torch.isfinite(token_level_rewards).all():
        raise ValueError("rewards must be finite and aligned with the response")
    if len(mask) < 2:
        raise ValueError("batch centering needs at least two independent prompts")
    if index is None or len(index) != len(mask) or len(set(index)) != len(index):
        raise ValueError("FlashREINFORCE requires one unique rollout uid per prompt")
    with torch.no_grad():
        rewards = token_level_rewards.float().sum(dim=-1)
        advantages = (rewards - rewards.mean()).unsqueeze(-1) * mask
    # veRL uses returns for diagnostics only; preserve raw, undiscounted returns.
    return advantages, rewards.unsqueeze(-1) * mask


def flashreinforce_loss(log_prob, behavior_log_prob, advantages, response_mask, *,
                       sequence_kl_threshold=1e-3, global_batch_size=None, dp_size=1):
    """One decision per full trajectory, with original B and T_i denominators.

    Gate, baseline and behavior logps have no gradient. The direct clamped-ratio
    surrogate follows PDF Appendix A: gradients are zero outside [-30, 30].
    Observation/padding values are replaced before arithmetic, not multiplied by
    zero after exp/NaN. A rejected trajectory keeps its place in the denominator.
    """
    threshold = float(sequence_kl_threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("sequence_kl_threshold must be finite and nonnegative")
    mask = _mask(response_mask)
    for name, value in (("learner logps", log_prob), ("behavior logps", behavior_log_prob),
                        ("advantages", advantages)):
        if value.shape != mask.shape or not torch.isfinite(value[mask]).all():
            raise ValueError(f"{name} must be finite on every policy token and aligned")
    if torch.any(log_prob[mask] > 0) or torch.any(behavior_log_prob[mask] > 0):
        raise ValueError("log probabilities cannot be positive")
    learner = torch.where(mask, log_prob.float(), 0.0)
    behavior = torch.where(mask, behavior_log_prob.detach().float(), 0.0)
    adv = torch.where(mask, advantages.detach().float(), 0.0)
    lengths = mask.sum(dim=-1)
    row_adv = adv.sum(dim=-1) / lengths
    if not torch.allclose(adv, row_adv.unsqueeze(-1) * mask, rtol=1e-5, atol=1e-7):
        raise ValueError("each trajectory must carry one constant batch-centered advantage")
    with torch.no_grad():
        p = behavior.exp().clamp(PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        q = learner.detach().exp().clamp(PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        divergence = p * (p.log() - q.log()) + (1 - p) * (torch.log1p(-p) - torch.log1p(-q))
        sequence_kl = torch.where(mask, divergence.clamp_min(0), 0.0).sum(-1) / lengths
        admitted = sequence_kl <= threshold
    ratio = (learner - behavior).clamp(-LOG_RATIO_LIMIT, LOG_RATIO_LIMIT).exp()
    per_sequence = -torch.where(mask, ratio * adv, 0.0).sum(-1) / lengths
    denominator = len(mask) if global_batch_size is None else float(global_batch_size)
    if not math.isfinite(denominator) or denominator < len(mask) or int(dp_size) != 1:
        raise ValueError("the current FlashREINFORCE adapter requires single-GPU full-batch normalization")
    loss = (per_sequence * admitted).sum() / denominator
    metrics = {
        "flashreinforce/sequence_kl_mean": sequence_kl.mean().item(),
        "flashreinforce/admitted_fraction": admitted.float().mean().item(),
        "flashreinforce/ratio_mean": ratio.detach()[mask].mean().item(),
        "flashreinforce/log_ratio_guard_fraction": (
            ((learner.detach() - behavior).abs() > LOG_RATIO_LIMIT)[mask].float().mean().item()
        ),
    }
    return loss, metrics


def compute_flashreinforce_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                                      loss_agg_mode="seq-mean-token-mean", config=None,
                                      rollout_is_weights=None):
    """veRL policy-loss registry interface; old_log_prob must be the sampler logps."""
    if loss_agg_mode != "seq-mean-token-mean" or rollout_is_weights is not None:
        raise ValueError("FlashREINFORCE supplies its own sequence gate, ratio and sample mean")
    if config is None:
        raise ValueError("FlashREINFORCE actor configuration is required")
    global_info = config.get("global_batch_info", {})
    return flashreinforce_loss(
        log_prob, old_log_prob, advantages, response_mask,
        sequence_kl_threshold=config["policy_loss"]["sequence_kl_threshold"],
        global_batch_size=global_info.get("global_batch_size"),
        dp_size=global_info.get("dp_size", 1),
    )


def validate_flashreinforce_config(config):
    """Reject changes that would silently turn this arm into another algorithm."""
    method = config.get("shopping_flashreinforce", {})
    algorithm = config["algorithm"]
    actor = config["actor_rollout_ref"]["actor"]
    loss = actor.get("policy_loss", {})
    selected = str(algorithm["adv_estimator"]) == METHOD or loss.get("loss_mode") == METHOD
    if not method.get("enable", False) and not selected:
        return
    if not method.get("enable", False) or not selected:
        raise ValueError("FlashREINFORCE method flag, estimator and loss must agree")
    from shopping_grpo.training.grpo.flash_batching import validate_admission_config
    validate_admission_config(config)
    from shopping_grpo.training.grpo.feedback_sampling import validate_feedback_config
    validate_feedback_config(config)
    execution = method.get("execution", "sync")
    if execution not in ("sync", "async"):
        raise ValueError("unknown FlashREINFORCE execution mode")
    if execution == "async":
        if method.get("invalid_batch_policy") != "skip_batch":
            raise ValueError("Shop async FlashREINFORCE requires explicit whole-batch admission")
        for key in ("async_inflight", "async_queue_batches", "async_timeout_seconds"):
            if isinstance(method.get(key), bool) or not isinstance(method.get(key), int) or method[key] <= 0:
                raise ValueError(f"shopping_flashreinforce.{key} must be a positive integer")
        if method["async_inflight"] <= int(config["data"]["train_batch_size"]):
            raise ValueError("async inflight pool must exceed one learner batch")
        rollout = config["actor_rollout_ref"]["rollout"]
        if rollout.get("free_cache_engine", True):
            raise ValueError("async sampling must stay resident during actor updates")
        if config["data"].get("dataloader_num_workers", 0) != 0:
            raise ValueError("async producer requires dataloader_num_workers=0 to avoid forking CUDA threads")
        if rollout.get("checkpoint_manager_class") != "shopping_grpo.training.grpo.async_runtime.AsyncCheckpointManager":
            raise ValueError("async sampling requires version-safe checkpoint manager")
        if rollout.get("agent", {}).get("agent_loop_manager_class") != "shopping_grpo.training.grpo.async_runtime.AsyncAgentLoopManager":
            raise ValueError("async sampling requires the partial-rollout client")
    rollout = config["actor_rollout_ref"]["rollout"]
    if str(algorithm["adv_estimator"]) != METHOD or loss.get("loss_mode") != METHOD:
        raise ValueError("FlashREINFORCE requires its own advantage estimator and policy loss")
    if int(rollout["n"]) != 1:
        raise ValueError("FlashREINFORCE requires rollout.n=1")
    batch_size = int(config["data"]["train_batch_size"])
    if batch_size < 2 or int(actor["ppo_mini_batch_size"]) != batch_size or int(actor["ppo_epochs"]) != 1:
        raise ValueError("FlashREINFORCE requires one full-batch optimizer step without reuse")
    if actor.get("use_dynamic_bsz", False) or int(actor["ppo_micro_batch_size_per_gpu"]) != 1:
        raise ValueError("FlashREINFORCE currently uses one complete trajectory per microbatch")
    if int(config["trainer"]["n_gpus_per_node"]) != 1 or int(config["trainer"]["nnodes"]) != 1:
        raise ValueError("FlashREINFORCE currently supports the single-GPU Shop recipe")
    if actor["loss_agg_mode"] != "seq-mean-token-mean":
        raise ValueError("FlashREINFORCE requires sample-mean loss")
    if config.get("shopping_dynamic_sampling", {}).get("enable", False):
        raise ValueError("FlashREINFORCE cannot use GRPO group filtering or resampling")
    if any(config.get(name, {}).get("enable", False) for name in ("shopping_gigpo", "shopping_trace")):
        raise ValueError("FlashREINFORCE must be compared separately from GiGPO and TRACE")
    if algorithm.get("use_kl_in_reward", False) or actor.get("use_kl_loss", False):
        raise ValueError("FlashREINFORCE does not use a reference KL term")
    if config.get("critic", {}).get("enable", False) or config["trainer"].get("critic_warmup", 0):
        raise ValueError("FlashREINFORCE is critic-free and has no critic warmup")
    if config.get("distillation", {}).get("enabled", False) or actor.get("entropy_coeff", 0) != 0:
        raise ValueError("FlashREINFORCE cannot include auxiliary distillation or entropy loss")
    correction = algorithm.get("rollout_correction", {})
    if not correction.get("bypass_mode", False) or not rollout.get("calculate_log_probs", False):
        raise ValueError("FlashREINFORCE requires actual rollout logps through bypass mode")
    if correction.get("rollout_is") is not None or correction.get("rollout_rs") is not None:
        raise ValueError("FlashREINFORCE supplies its own IS and sequence rejection")
    # Processed sampler logps and learner scores must describe the same
    # full-support policy. No nucleus/top-k truncation or sampling penalties.
    if float(rollout["temperature"]) != 1 or float(rollout["top_p"]) != 1 or int(rollout["top_k"]) != -1:
        raise ValueError("FlashREINFORCE requires temperature=1, top_p=1 and top_k=-1")
    if rollout.get("logprobs_mode") != "processed_logprobs" or float(rollout.get("repetition_penalty", 1)) != 1:
        raise ValueError("FlashREINFORCE requires processed sampler logps without repetition penalties")
    threshold = float(loss["sequence_kl_threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("sequence_kl_threshold must be finite and nonnegative")


def validate_flashreinforce_batch(batch, config):
    """Keep invalid episodes visible by stopping the batch, never selecting survivors."""
    from shopping_grpo.training.grpo.dynamic_sampling import extract_shopping_group_signals

    validate_flashreinforce_config(config)
    mask = _mask(batch.batch["response_mask"])
    if len(mask) != int(config["data"]["train_batch_size"]):
        raise ValueError("incomplete FlashREINFORCE update batch")
    uids = list(batch.non_tensor_batch["uid"])
    infos = list(batch.non_tensor_batch["shopping"])
    if len(uids) != len(mask) or len(set(uids)) != len(mask) or len(infos) != len(mask):
        raise ValueError("FlashREINFORCE requires distinct aligned rollout identities")
    task_ids = [info["task_id"] for info in infos]
    if len(set(task_ids)) != len(mask):
        raise ValueError("FlashREINFORCE needs different tasks within a batch")
    _, _, invalid, reasons = extract_shopping_group_signals(infos)
    if any(invalid):
        raise ValueError(f"invalid FlashREINFORCE episodes; batch not updated: {reasons}")
    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        raise ValueError("missing actual sampler log probabilities")
    if not torch.equal(batch.batch["old_log_probs"][mask], batch.batch["rollout_log_probs"][mask]):
        raise ValueError("old_log_probs must retain actual sampler probabilities")
    rewards = batch.batch["token_level_rewards"].float().sum(-1)
    native = torch.tensor([info["reward"]["total"] for info in infos], device=rewards.device)
    if not torch.isfinite(rewards).all() or not torch.allclose(rewards, native, atol=1e-6, rtol=1e-6):
        raise ValueError("the first FlashREINFORCE comparison requires unmodified native rewards")
    expected, _ = compute_flashreinforce_advantage(batch.batch["token_level_rewards"], mask, uids)
    if not torch.allclose(batch.batch["advantages"].float(), expected):
        raise ValueError("FlashREINFORCE advantages must be centered over the full fresh batch")
    if config["shopping_flashreinforce"].get("execution", "sync") == "async":
        version = batch.meta_info.get("flash_learner_version")
        if not isinstance(version, int) or version < 0:
            raise ValueError("async learner version is missing")
        for row, info in enumerate(infos):
            versions = info.get("sampling_policy_versions", [])
            if len(versions) != int(mask[row].sum()) or any(
                not isinstance(v, int) or v < 0 or v > version for v in versions
            ):
                raise ValueError("async policy-token sampling provenance is incomplete or from the future")


def install_flashreinforce():
    """Register the optimizer in every Ray process via the existing setup hook."""
    from verl.trainer.ppo import core_algos, rollout_corr_helper
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if hasattr(RayPPOTrainer, "fit"):
        from shopping_grpo.training.grpo.flash_batching import install_batch_admission
        install_batch_admission(RayPPOTrainer)
    if hasattr(RayPPOTrainer, "init_workers"):
        from shopping_grpo.training.grpo.async_fit import install_worker_setup
        install_worker_setup(RayPPOTrainer)

    core_algos.register_adv_est(METHOD)(compute_flashreinforce_advantage)
    core_algos.register_policy_loss(METHOD)(compute_flashreinforce_policy_loss)
    original_bypass = rollout_corr_helper.apply_bypass_mode
    if not getattr(original_bypass, "_shopping_flashreinforce", False):
        @wraps(original_bypass)
        def apply_bypass(batch, rollout_corr_config=None, policy_loss_config=None):
            if policy_loss_config is not None and policy_loss_config.get("loss_mode") == METHOD:
                if "rollout_log_probs" not in batch.batch:
                    raise ValueError("FlashREINFORCE requires rollout_log_probs")
                batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
                return  # Retain the custom loss; native helper would overwrite it.
            return original_bypass(batch, rollout_corr_config, policy_loss_config)
        apply_bypass._shopping_flashreinforce = True
        rollout_corr_helper.apply_bypass_mode = apply_bypass

    original_update = RayPPOTrainer._update_actor
    if getattr(original_update, "_shopping_flashreinforce", False):
        return

    @wraps(original_update)
    def update_actor(self, batch):
        if not self.config.get("shopping_flashreinforce", {}).get("enable", False):
            return original_update(self, batch)
        validate_flashreinforce_batch(batch, self.config)
        uids = set(batch.non_tensor_batch["uid"])
        consumed = getattr(self, "_shopping_flashreinforce_consumed", set())
        if consumed & uids:
            raise ValueError("a FlashREINFORCE rollout batch cannot be updated twice")
        pool = getattr(self, "_shopping_flash_pool", None)
        before = pool.metrics()["flash_async/finished"] if pool is not None else 0
        output = original_update(self, batch)
        completed_during_update = pool.metrics()["flash_async/finished"] - before if pool is not None else 0
        self._shopping_flashreinforce_consumed = consumed | uids
        rewards = batch.batch["token_level_rewards"].sum(-1)
        output.meta_info["metrics"].update({
            "flashreinforce/batch_reward_mean": rewards.mean().item(),
            "flashreinforce/batch_reward_std": rewards.std(unbiased=False).item(),
            "flashreinforce/zero_advantage_batch": int(bool(torch.all(rewards == rewards[0]))),
            "flashreinforce/fresh_trajectories": len(uids),
            "flashreinforce/optimizer_steps_per_batch": 1,
            "flashreinforce/cross_update_async": int(self.config["shopping_flashreinforce"].get("execution", "sync") == "async"),
        })
        if self.config["shopping_flashreinforce"].get("execution", "sync") == "async":
            ages = [batch.meta_info["flash_learner_version"] - v
                    for info in batch.non_tensor_batch["shopping"] for v in info["sampling_policy_versions"]]
            output.meta_info["metrics"].update({
                "flash_async/completions_during_update": completed_during_update,
                "flash_async/policy_lag_min": min(ages), "flash_async/policy_lag_max": max(ages),
                "flash_async/policy_lag_mean": sum(ages) / len(ages),
                "flash_async/mixed_version_trajectories": sum(len(set(info["sampling_policy_versions"])) > 1
                    for info in batch.non_tensor_batch["shopping"]),
                "flash_async/sampling_interruptions": sum(info.get("sampling_interruptions", 0)
                    for info in batch.non_tensor_batch["shopping"]),
            })
        return output

    update_actor._shopping_flashreinforce = True
    RayPPOTrainer._update_actor = update_actor
