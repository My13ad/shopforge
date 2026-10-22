"""CPU math, reference parity, configuration and one-pass integration checks."""
import ast
from dataclasses import dataclass, field
from enum import Enum
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Optional

import pytest
import torch

from shopping_grpo.training.grpo import flashreinforce as flash
from test_grpo_entry_and_hook import args, entry, parent  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "outputs/preparation/v1/runtime-wheel/source/verl"
MOLT = ROOT / "references/flashreinforce-upstream/labs-molt-main/molt"


def config(batch_size=4):
    return {
        "shopping_flashreinforce": {"enable": True},
        "shopping_dynamic_sampling": {"enable": False},
        "shopping_gigpo": {"enable": False}, "shopping_trace": {"enable": False},
        "algorithm": {"adv_estimator": flash.METHOD, "use_kl_in_reward": False,
                      "rollout_correction": {"bypass_mode": True, "rollout_is": None, "rollout_rs": None}},
        "data": {"train_batch_size": batch_size},
        "critic": {"enable": False},
        "distillation": {"enabled": False},
        "trainer": {"n_gpus_per_node": 1, "nnodes": 1, "critic_warmup": 0},
        "actor_rollout_ref": {
            "rollout": {"n": 1, "calculate_log_probs": True, "temperature": 1., "top_p": 1., "top_k": -1,
                        "logprobs_mode": "processed_logprobs", "repetition_penalty": 1.},
            "actor": {"ppo_mini_batch_size": batch_size, "ppo_micro_batch_size_per_gpu": 1,
                      "ppo_epochs": 1, "loss_agg_mode": "seq-mean-token-mean", "use_dynamic_bsz": False,
                      "use_kl_loss": False, "entropy_coeff": 0.,
                      "policy_loss": {"loss_mode": flash.METHOD, "sequence_kl_threshold": 1e-3},
                      "global_batch_info": {"dp_size": 1, "global_batch_size": batch_size}},
        },
    }


def async_config(settings):
    size=settings['data']['train_batch_size']
    settings['shopping_flashreinforce'].update(execution='async',invalid_batch_policy='skip_batch',
        max_candidate_batches=256,max_consecutive_skipped_batches=16,
        async_inflight=4*size,async_queue_batches=8,async_timeout_seconds=3)
    settings['trainer']['resume_mode']='disable'
    settings['data']['dataloader_num_workers']=0
    settings['actor_rollout_ref']['rollout'].update(free_cache_engine=False,
        checkpoint_manager_class='shopping_grpo.training.grpo.async_runtime.AsyncCheckpointManager',
        agent={'agent_loop_manager_class':'shopping_grpo.training.grpo.async_runtime.AsyncAgentLoopManager'})
    return settings


def batch():
    mask = torch.tensor([[1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]])
    rewards = torch.zeros(4, 4)
    rewards[:, -1] = torch.tensor([1., .55, -.15, -.85])
    uids = [f"fresh-{i}" for i in range(4)]
    adv, returns = flash.compute_flashreinforce_advantage(rewards, mask, uids)
    sampler = torch.full((4, 4), math.log(.5))
    infos = [{"task_id": i, "infrastructure_invalid": False,
              "reward": {"total": r, "terminal_utility": r, "purchase_success": r > 0,
                         "sampling_invalid": False}}
             for i, r in enumerate([1., .55, -.15, -.85])]
    return SimpleNamespace(
        batch={"response_mask": mask, "token_level_rewards": rewards,
               "rollout_log_probs": sampler, "old_log_probs": sampler.clone(),
               "advantages": adv, "returns": returns},
        non_tensor_batch={"uid": uids, "shopping": infos}, meta_info={})


def test_batch_centering_scalar_rewards_signed_feedback_no_whitening():
    data = batch()
    mask = data.batch["response_mask"].bool()
    expected = torch.tensor([.8625, .4125, -.2875, -.9875])
    torch.testing.assert_close(data.batch["advantages"], expected[:, None] * mask)
    assert not data.batch["advantages"].requires_grad
    torch.testing.assert_close(data.batch["returns"][:, 0], torch.tensor([1., .55, -.15, -.85]))


def test_equal_rewards_have_zero_gradient_and_are_not_resampled():
    mask = torch.ones(4, 3)
    rewards = torch.zeros_like(mask)
    rewards[:, -1] = .55
    adv, _ = flash.compute_flashreinforce_advantage(rewards, mask, range(4))
    learner = torch.full_like(mask, -1., requires_grad=True)
    loss, metrics = flash.flashreinforce_loss(learner, learner.detach(), adv, mask)
    loss.backward()
    assert loss.item() == 0 and learner.grad.count_nonzero() == 0
    assert metrics["flashreinforce/admitted_fraction"] == 1


def test_length_normalization_and_masked_observations_keep_equal_trajectory_weight():
    data = batch().batch
    learner = data["old_log_probs"].clone().requires_grad_()
    loss, _ = flash.flashreinforce_loss(learner, data["old_log_probs"], data["advantages"], data["response_mask"])
    loss.backward()
    lengths = data["response_mask"].sum(-1)
    expected = -data["advantages"] / lengths[:, None] / 4
    torch.testing.assert_close(learner.grad, expected)
    torch.testing.assert_close(learner.grad.sum(-1), -data["advantages"][:, 0] / 4)


def test_sequence_rejection_keeps_original_batch_and_length_denominators():
    learner = torch.tensor([[math.log(.99), math.log(.5)], [math.log(.5), -1.]], requires_grad=True)
    behavior = torch.full((2, 2), math.log(.5))
    mask = torch.tensor([[1, 1], [1, 0]])
    adv = torch.tensor([[1., 1.], [-1., 0.]])
    loss, metrics = flash.flashreinforce_loss(learner, behavior, adv, mask, sequence_kl_threshold=.01)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(.5))
    torch.testing.assert_close(learner.grad, torch.tensor([[0., 0.], [.5, 0.]]))
    assert metrics["flashreinforce/admitted_fraction"] == .5


def test_boundary_is_inclusive_and_complete_rejection_has_finite_zero_gradient():
    learner = torch.tensor([[math.log(.8), math.log(.7)]], requires_grad=True)
    behavior = torch.full_like(learner, math.log(.5))
    adv, mask = torch.ones_like(learner), torch.ones_like(learner)
    _, stats = flash.flashreinforce_loss(learner, behavior, adv, mask, sequence_kl_threshold=1)
    exact = stats["flashreinforce/sequence_kl_mean"]
    _, boundary = flash.flashreinforce_loss(learner, behavior, adv, mask, sequence_kl_threshold=exact)
    assert boundary["flashreinforce/admitted_fraction"] == 1
    rejected, _ = flash.flashreinforce_loss(learner, behavior, adv, mask, sequence_kl_threshold=0)
    rejected.backward()
    assert rejected.item() == 0 and learner.grad.count_nonzero() == 0


def test_unclipped_is_gradient_and_stop_gradient_behavior_advantage():
    learner = torch.tensor([[math.log(.6)]], requires_grad=True)
    behavior = torch.tensor([[math.log(.2)]], requires_grad=True)
    advantage = torch.tensor([[2.]], requires_grad=True)
    loss, _ = flash.flashreinforce_loss(learner, behavior, advantage, torch.ones(1, 1), sequence_kl_threshold=10)
    loss.backward()
    torch.testing.assert_close(learner.grad, torch.tensor([[-6.]]))
    assert behavior.grad is None and advantage.grad is None


@pytest.mark.parametrize("learner,behavior", [(-1., -100.), (-100., -1.)])
def test_pdf_appendix_a_log_ratio_guard_stops_out_of_range_gradient(learner, behavior):
    lp = torch.tensor([[learner]], requires_grad=True)
    loss, metrics = flash.flashreinforce_loss(lp, torch.tensor([[behavior]]), torch.ones(1, 1),
                                            torch.ones(1, 1), sequence_kl_threshold=100)
    loss.backward()
    assert torch.isfinite(loss) and lp.grad.item() == 0
    assert metrics["flashreinforce/log_ratio_guard_fraction"] == 1


def test_masked_nan_and_inf_do_not_enter_gate_ratio_or_gradient():
    learner = torch.tensor([[-1., float("nan"), -1., float("inf")]], requires_grad=True)
    behavior = torch.tensor([[-1., float("inf"), -1., float("nan")]])
    loss, metrics = flash.flashreinforce_loss(learner, behavior, torch.tensor([[1., float("nan"), 1., 0.]]),
                                            torch.tensor([[1, 0, 1, 0]]))
    loss.backward()
    torch.testing.assert_close(learner.grad, torch.tensor([[-.5, 0., -.5, 0.]]))
    assert loss.item() == -1 and metrics["flashreinforce/sequence_kl_mean"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), .1])
def test_bad_active_log_prob_is_rejected(bad):
    with pytest.raises(ValueError):
        flash.flashreinforce_loss(torch.tensor([[bad]]), torch.tensor([[-1.]]), torch.ones(1, 1), torch.ones(1, 1))


def test_microbatch_accumulation_equals_full_batch_loss_and_gradient():
    data = batch().batch
    full = data["old_log_probs"].clone().requires_grad_()
    loss, _ = flash.flashreinforce_loss(full, data["old_log_probs"], data["advantages"], data["response_mask"])
    loss.backward()
    micro = full.detach().clone().requires_grad_()
    accumulated = 0
    for i in range(4):
        part, _ = flash.compute_flashreinforce_policy_loss(
            data["old_log_probs"][i:i+1], micro[i:i+1], data["advantages"][i:i+1],
            data["response_mask"][i:i+1], config=config()["actor_rollout_ref"]["actor"])
        accumulated += part.item()
        part.backward()
    torch.testing.assert_close(micro.grad, full.grad)
    assert accumulated == pytest.approx(loss.item(), abs=1e-7)


@pytest.mark.parametrize("path,value", [
    (("actor_rollout_ref", "rollout", "n"), 4),
    (("actor_rollout_ref", "rollout", "calculate_log_probs"), False),
    (("actor_rollout_ref", "rollout", "top_p"), .9),
    (("actor_rollout_ref", "rollout", "logprobs_mode"), "raw_logprobs"),
    (("actor_rollout_ref", "rollout", "repetition_penalty"), 1.1),
    (("actor_rollout_ref", "actor", "ppo_epochs"), 2),
    (("actor_rollout_ref", "actor", "ppo_mini_batch_size"), 2),
    (("actor_rollout_ref", "actor", "loss_agg_mode"), "token-mean"),
    (("actor_rollout_ref", "actor", "policy_loss", "loss_mode"), "bypass_mode"),
    (("algorithm", "rollout_correction", "bypass_mode"), False),
    (("algorithm", "rollout_correction", "rollout_is"), "token"),
    (("algorithm", "adv_estimator"), "grpo"),
    (("shopping_dynamic_sampling", "enable"), True),
    (("shopping_gigpo", "enable"), True),
    (("shopping_trace", "enable"), True),
    (("critic", "enable"), True),
    (("trainer", "n_gpus_per_node"), 2),
    (("shopping_flashreinforce", "enable"), False),
    (("distillation", "enabled"), True),
])
def test_configuration_cannot_silently_change_algorithm(path, value):
    settings = config()
    cursor = settings
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError):
        flash.validate_flashreinforce_config(settings)


@pytest.mark.parametrize("mutation", ["duplicate_task", "duplicate_uid", "invalid", "old_logps", "reward", "advantage"])
def test_batch_integrity_stops_without_filtering_or_mutating(mutation):
    data = batch()
    if mutation == "duplicate_task":
        data.non_tensor_batch["shopping"][0]["task_id"] = 1
    elif mutation == "duplicate_uid":
        data.non_tensor_batch["uid"][0] = data.non_tensor_batch["uid"][1]
    elif mutation == "invalid":
        data.non_tensor_batch["shopping"][0]["infrastructure_invalid"] = True
    elif mutation == "old_logps":
        data.batch["old_log_probs"][0, 0] -= .2
    elif mutation == "reward":
        data.batch["token_level_rewards"][0, -1] -= .2
    else:
        data.batch["advantages"][0, 0] += .2
    before = data.batch["response_mask"].clone()
    with pytest.raises(ValueError):
        flash.validate_flashreinforce_batch(data, config())
    assert torch.equal(before, data.batch["response_mask"])


def test_entry_builds_flash_probe_and_main_without_launch(monkeypatch, parent, tmp_path):
    for profile in ("smoke", "probe", "main"):
        command, environment = entry.build_command(args(monkeypatch, parent, tmp_path, profile=profile, method=flash.METHOD))
        assert "algorithm.adv_estimator=flashreinforce" in command
        assert "actor_rollout_ref.actor.ppo_mini_batch_size=8" in command
        assert "data.train_batch_size=8" in command
        assert "shopping_dynamic_sampling.enable=false" in command
        assert environment["SHOPPING_LENGTH_SHAPING_ENABLE"] == "false"
        assert environment["GRPO_TRAIN_FILE"].endswith("smoke8.parquet" if profile == "smoke" else "train953.parquet")
        if profile == "probe":
            assert "actor_rollout_ref.rollout.val_kwargs.n=1" in command
            assert environment["GRPO_VAL_FILE"].endswith("probe60.parquet")
        else:
            assert "actor_rollout_ref.rollout.val_kwargs.n=4" not in command
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("policy", ["skip_batch", "error"])
def test_entry_records_explicit_admission_policy_without_running(monkeypatch, parent, tmp_path, capsys, policy):
    import json
    options = args(monkeypatch, parent, tmp_path, profile="main", method=flash.METHOD,
                   flash_invalid_batch_policy=policy)
    options.dry_run = True
    monkeypatch.setattr(entry, "parse_args", lambda: options)
    entry.main()
    audit = json.loads(capsys.readouterr().out)
    assert audit["flashreinforce"]["invalid_batch_policy"] == policy
    assert f"shopping_flashreinforce.invalid_batch_policy={policy}" in audit["command"]
    assert not (tmp_path / "output").exists()


def test_entry_refuses_hidden_admission_policy_override(monkeypatch, parent, tmp_path):
    options = args(monkeypatch, parent, tmp_path, profile="main", method=flash.METHOD)
    options.hydra_overrides = ["--", "++shopping_flashreinforce.invalid_batch_policy=error"]
    with pytest.raises(SystemExit, match="--flash-invalid-batch-policy"):
        entry.build_command(options)


def _source_nodes(path, names, namespace):
    if not path.is_file():
        pytest.skip(f"downloaded reference not available: {path.name}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize('execution',['sync','async'])
def test_actual_pinned_verl_registry_bypass_and_one_cpu_optimizer_step(monkeypatch,execution):
    """Execute pinned veRL dispatch/helper functions without loading Ray/CUDA."""
    core = ModuleType("verl.trainer.ppo.core_algos")
    core.__dict__.update({"Any": Any, "Callable": Callable, "Enum": Enum,
                          "PolicyLossFn": Callable,
                          "POLICY_LOSS_REGISTRY": {}, "ADV_ESTIMATOR_REGISTRY": {}})
    _source_nodes(UPSTREAM / "trainer/ppo/core_algos.py", ["AdvantageEstimator", "register_adv_est",
        "register_policy_loss", "get_adv_estimator_fn", "get_policy_loss_fn"], core.__dict__)
    helper = ModuleType("verl.trainer.ppo.rollout_corr_helper")
    helper.__dict__.update({"DataProto": Any, "Optional": Optional, "RolloutCorrectionConfig": Any,
                            "PolicyLossConfig": Any})
    _source_nodes(UPSTREAM / "trainer/ppo/rollout_corr_helper.py", ["apply_bypass_mode"], helper.__dict__)
    native_bypass = helper.apply_bypass_mode
    native = ModuleType("verl.trainer.ppo.ray_trainer")
    native.__dict__.update({"DataProto": Any, "AdvantageEstimator": core.AdvantageEstimator,
                           "Optional": Optional, "AlgoConfig": Any, "core_algos": core})
    _source_nodes(UPSTREAM / "trainer/ppo/ray_trainer.py", ["compute_advantage"], native.__dict__)
    parameter = torch.nn.Parameter(torch.zeros(4, 4))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=0)
    steps = []

    class Trainer:
        def _update_actor(self, data):
            optimizer.zero_grad()
            logps = torch.nn.functional.logsigmoid(parameter)
            for row in range(4):
                loss, _ = core.get_policy_loss_fn(flash.METHOD)(
                    data.batch["old_log_probs"][row:row+1], logps[row:row+1],
                    data.batch["advantages"][row:row+1], data.batch["response_mask"][row:row+1],
                    config=self.config["actor_rollout_ref"]["actor"])
                loss.backward(retain_graph=True)
            optimizer.step()
            steps.append(1)
            return SimpleNamespace(meta_info={"metrics": {}})

    native.RayPPOTrainer = Trainer
    package = ModuleType("verl.trainer.ppo")
    package.core_algos, package.rollout_corr_helper = core, helper
    for name, module in (("verl", ModuleType("verl")), ("verl.trainer", ModuleType("verl.trainer")),
                         ("verl.trainer.ppo", package), (core.__name__, core),
                         (helper.__name__, helper), (native.__name__, native)):
        monkeypatch.setitem(sys.modules, name, module)
    flash.install_flashreinforce()
    flash.install_flashreinforce()
    trainer = Trainer()
    trainer.config = config()
    data = batch()
    if execution=='async':
        async_config(trainer.config)
        data.meta_info['flash_learner_version']=2
        for row,info in enumerate(data.non_tensor_batch['shopping']):
            info['sampling_policy_versions']=[0]*int(data.batch['response_mask'][row].sum())
    before = parameter.detach().clone()
    helper.apply_bypass_mode(data, policy_loss_config=trainer.config["actor_rollout_ref"]["actor"]["policy_loss"])
    assert helper.apply_bypass_mode is not native_bypass
    assert trainer.config["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] == flash.METHOD
    data = native.compute_advantage(data, flash.METHOD, config=trainer.config["algorithm"])
    output = trainer._update_actor(data)
    assert len(steps) == 1 and not torch.equal(before, parameter.detach())
    assert optimizer.state[parameter]["step"].item() == 1
    assert output.meta_info["metrics"]["flashreinforce/optimizer_steps_per_batch"] == 1
    assert output.meta_info['metrics']['flashreinforce/cross_update_async']==int(execution=='async')
    if execution=='async':
        assert output.meta_info['metrics']['flash_async/policy_lag_max']==2
    with pytest.raises(ValueError, match="twice"):
        trainer._update_actor(data)
    assert len(steps) == 1
    from omegaconf import OmegaConf
    grpo_loss = OmegaConf.create({"loss_mode": "vanilla", "rollout_correction": {}})
    helper.apply_bypass_mode(batch(), policy_loss_config=grpo_loss)
    assert grpo_loss.loss_mode == "bypass_mode"


def test_real_hydra_recipe_and_typed_policy_config(monkeypatch, parent, tmp_path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # Execute real veRL dataclasses without importing its distributed runtime.
    spec = importlib.util.spec_from_file_location("flash_native_base", UPSTREAM / "base_config.py")
    if not Path(spec.origin).is_file():
        pytest.skip("pinned veRL sources unavailable")
    base = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, base)
    spec.loader.exec_module(base)
    typed = ModuleType("flash_native_typed")
    monkeypatch.setitem(sys.modules, typed.__name__, typed)
    typed.__dict__.update({"dataclass": dataclass, "field": field, "BaseConfig": base.BaseConfig,
                           "Optional": Optional, "Any": Any})
    _source_nodes(UPSTREAM / "trainer/config/algorithm.py", ["RolloutCorrectionConfig"], typed.__dict__)
    _source_nodes(UPSTREAM / "workers/config/actor.py", ["PolicyLossConfig"], typed.__dict__)
    workers = ModuleType("verl.workers.config")
    workers.PolicyLossConfig = typed.PolicyLossConfig
    monkeypatch.setitem(sys.modules, "verl", ModuleType("verl"))
    monkeypatch.setitem(sys.modules, "verl.workers", ModuleType("verl.workers"))
    monkeypatch.setitem(sys.modules, workers.__name__, workers)
    for profile in ("smoke", "probe", "main"):
        command, environment = entry.build_command(args(monkeypatch, parent, tmp_path, profile=profile, method=flash.METHOD))
        for key, value in environment.items():
            monkeypatch.setenv(key, value)
        search = (UPSTREAM / "trainer/config").as_posix()
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            recipe = compose(config_name="grpo", overrides=command[5:] + [f'hydra.searchpath=["file://{search}"]'])
        flash.validate_flashreinforce_config(recipe)
        actor_loss = instantiate(recipe.actor_rollout_ref.actor.policy_loss, _convert_="partial")
        assert isinstance(actor_loss, typed.PolicyLossConfig)
        assert actor_loss.sequence_kl_threshold == 1e-3
        assert actor_loss.loss_mode == flash.METHOD
        assert recipe.actor_rollout_ref.rollout.n == 1
        assert recipe.actor_rollout_ref.rollout.val_kwargs.n == 1
        assert recipe.trainer.total_training_steps == (64 if profile == "main" else 1)
        assert OmegaConf.to_container(recipe, resolve=True)["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 8


def test_downloaded_molt_matches_loss_and_gradient_inside_numerical_guard(monkeypatch):
    if not (MOLT / "models/loss.py").is_file():
        pytest.skip("optional NVIDIA reference checkout unavailable")
    package = ModuleType("flash_reference")
    package.__path__ = [str(MOLT / "models")]
    monkeypatch.setitem(sys.modules, "flash_reference", package)
    spec = importlib.util.spec_from_file_location("flash_reference.loss", MOLT / "models/loss.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = batch().batch
    behavior = data["old_log_probs"]
    learner = (behavior + torch.tensor([[.4, 0, 0, 0], [.005, 0, -.005, 0],
                                        [0, 0, 0, 0], [.004, -.004, 0, 0]])).requires_grad_()
    ours, _ = flash.flashreinforce_loss(learner, behavior, data["advantages"], data["response_mask"])
    gradient = torch.autograd.grad(ours, learner)[0]
    reference, *_ = module.PolicyLoss(
        is_correction_level="seq", is_correction_gating="binary_kl",
        is_correction_threshold=[-float("inf"), 1e-3], loss_agg_mode="seq-mean-token-mean",
    )(learner, learner.detach(), data["advantages"], action_mask=data["response_mask"], rollout_log_probs=behavior)
    torch.testing.assert_close(ours, reference)
    torch.testing.assert_close(gradient, torch.autograd.grad(reference, learner)[0])
