"""Drive the pinned native fit loop with real CPU AdamW and a fake rollout worker."""
import ast
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import uuid

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from shopping_grpo.training.grpo import flashreinforce as flash
from test_flashreinforce import batch, config, async_config
from test_validation_unpadding import patched_source


class Tensors(dict):
    def __len__(self):
        return next(iter(self.values())).shape[0]


class Proto:
    def __init__(self, data):
        self.batch = Tensors(deepcopy(data.batch))
        self.non_tensor_batch = {k: np.array(v, dtype=object) for k, v in data.non_tensor_batch.items()}
        self.meta_info = deepcopy(data.meta_info)

    @staticmethod
    def from_single_dict(data):
        return Proto(data)

    def __len__(self):
        return len(self.batch)

    def select_idxs(self, indices):
        return Proto(SimpleNamespace(batch={k:v[indices] for k,v in self.batch.items()},
            non_tensor_batch={k:v[indices] for k,v in self.non_tensor_batch.items()},meta_info=dict(self.meta_info)))

    @staticmethod
    def concat(items):
        return Proto(SimpleNamespace(batch={k:torch.cat([p.batch[k] for p in items]) for k in items[0].batch},
            non_tensor_batch={k:np.concatenate([p.non_tensor_batch[k] for p in items]) for k in items[0].non_tensor_batch},
            meta_info=dict(items[0].meta_info)))

    def repeat(self, **kwargs):
        assert kwargs['repeat_times'] == 1
        return self

    def slice(self, start, stop):
        assert start == 0 and stop == len(self)
        return self

    def union(self, other):
        self.batch.update(other.batch)
        self.non_tensor_batch.update({k: v for k, v in other.non_tensor_batch.items() if k != 'uid'})
        return self


def candidate(invalid=False, equal=False):
    data = batch()
    data.batch['attention_mask'] = torch.ones(4, 4)
    data.non_tensor_batch['multi_modal_inputs'] = [{} for _ in range(4)]
    if invalid:
        data.non_tensor_batch['shopping'][0]['reward']['sampling_invalid'] = True
        data.non_tensor_batch['shopping'][0]['termination_reason'] = 'assistant_no_tool_call'
        # Even a bad episode with no policy tokens must be diagnosed and skipped.
        data.batch['response_mask'][0] = 0
    if equal:
        data.batch['token_level_rewards'][:] = 0
        for info in data.non_tensor_batch['shopping']:
            info['reward']['total'] = info['reward']['terminal_utility'] = 0.
    return data


def native_harness(monkeypatch, tmp_path, candidates, *, target=2, policy='skip_batch', limit=20, consecutive=3, execution='sync'):
    settings = config()
    settings['shopping_flashreinforce'].update(invalid_batch_policy=policy,
        max_candidate_batches=limit, max_consecutive_skipped_batches=consecutive)
    settings['trainer'].update(total_epochs=1, resume_mode='disable', project_name='cpu',
        experiment_name='cpu', logger=['console'], val_before_train=False, balance_batch=False,
        save_freq=1, test_freq=1, esi_redundant_time=0, default_local_dir=str(tmp_path/'checkpoints'))
    settings['algorithm'].update(gamma=1., lam=1.)
    settings['global_profiler'] = {'steps': None, 'profile_continuous_steps': False}
    settings['actor_rollout_ref']['rollout']['skip'] = {'enable': False}
    if execution=='async':
        async_config(settings)
    recipe = OmegaConf.create(settings)
    events, weights, sleeps, saves, validations, observed = [], [], [], [], [], []
    tracking = ModuleType('verl.utils.tracking')
    tracking.Tracking = lambda **kwargs: SimpleNamespace(log=lambda **kwargs: events.append(kwargs))
    helper = ModuleType('verl.trainer.ppo.rollout_corr_helper')
    helper.apply_bypass_mode = lambda batch, **kwargs: batch.batch.update(old_log_probs=batch.batch['rollout_log_probs'])
    for name in ('verl', 'verl.utils', 'verl.trainer', 'verl.trainer.ppo'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, tracking.__name__, tracking)
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    import shopping_grpo.training.grpo.dynamic_sampling as dynamic
    monkeypatch.setattr(dynamic, 'aggregate_shopping_metrics', lambda infos: {})
    diagnostic_path = tmp_path/'diagnostics.jsonl'
    monkeypatch.setenv('SHOPPING_GRPO_DIAGNOSTICS_PATH', str(diagnostic_path))

    @contextmanager
    def timer(name, timings, **kwargs):
        yield
        timings[name] = .001

    def advantage(data, **kwargs):
        data.batch['advantages'], data.batch['returns'] = flash.compute_flashreinforce_advantage(
            data.batch['token_level_rewards'], data.batch['response_mask'], data.non_tensor_batch['uid'])
        return data

    namespace = dict(os=os, np=np, uuid=uuid, torch=torch, DataProto=Proto, marked_timer=timer,
        AdvantageEstimator=SimpleNamespace(REMAX='remax', GRPO='grpo', GDPO='gdpo'),
        compute_advantage=advantage, extract_reward=lambda data: (data.batch['token_level_rewards'], {}),
        should_save_ckpt_esi=lambda **kwargs: False, reduce_metrics=lambda metrics: metrics,
        tqdm=lambda **kwargs: SimpleNamespace(update=lambda n: None, close=lambda: None), pprint=lambda *args: None)
    for name in ('compute_data_metrics', 'compute_timing_metrics', 'compute_throughout_metrics',
                 'compute_variance_proxy_metrics', 'compute_spec_decode_metrics'):
        namespace[name] = lambda *args, **kwargs: {}
    source = patched_source()
    source_file = tmp_path/'pinned_ray_trainer.py'
    source_file.write_text(source, encoding='utf-8')
    tree = ast.parse(source)
    trainer_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    method = next(n for n in trainer_node.body if getattr(n, 'name', None) == 'fit')
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_file), 'exec'), namespace)
    fit = namespace['fit']
    parameter = torch.nn.Parameter(torch.zeros(4, 4))
    optimizer = torch.optim.AdamW([parameter], lr=.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    rollout_calls = []

    def generate(data):
        rollout_calls.append(len(rollout_calls)+1)
        data.meta_info['timing'] = {}
        return data

    class Trainer:
        def _update_actor(self, data):
            flash.validate_flashreinforce_batch(data, self.config)
            observed.append(deepcopy(data))
            optimizer.zero_grad()
            logps = torch.nn.functional.logsigmoid(parameter)
            loss, _ = flash.flashreinforce_loss(logps, data.batch['old_log_probs'],
                data.batch['advantages'], data.batch['response_mask'])
            loss.backward()
            optimizer.step()
            scheduler.step()
            return SimpleNamespace(meta_info={'metrics': {'actor/grad_norm': 1.}})

        def _save_checkpoint(self):
            saves.append(self.global_steps)
            path = Path(self.config.trainer.default_local_dir)/f'global_step_{self.global_steps}'
            path.mkdir(parents=True)

    Trainer.fit = fit
    from shopping_grpo.training.grpo.flash_batching import install_batch_admission
    install_batch_admission(Trainer)
    installed = Trainer.fit
    install_batch_admission(Trainer)
    assert Trainer.fit is installed
    trainer = Trainer()
    trainer.config = recipe
    trainer.total_training_steps = target
    trainer._dump_executor = SimpleNamespace(_shutdown=False)
    trainer._load_checkpoint = lambda: None
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: weights.append(step),
        sleep_replicas=lambda: sleeps.append(trainer.global_steps))
    trainer._get_gen_batch = lambda data: data
    trainer._start_profiling = trainer._stop_profiling = lambda *args: None
    trainer._shutdown_dump_executor = lambda: None
    trainer._validate = lambda: (validations.append(trainer.global_steps) or {'validation': 1.})
    trainer.train_dataloader = candidates
    trainer.train_dataset = object()
    trainer.actor_rollout_wg = object()
    trainer.async_rollout_manager = SimpleNamespace(generate_sequences=generate)
    if execution=='async':
        from concurrent.futures import Future
        ray=ModuleType('ray')
        monkeypatch.setitem(sys.modules,'ray',ray)
        monkeypatch.setattr(ray,'cancel',lambda ref,force=False:ref.cancel(),raising=False)
        monkeypatch.setattr(sys.modules['verl'],'DataProto',Proto,raising=False)
        trainer.checkpoint_manager.sampling_version=0
        def update(step):
            weights.append(step);trainer.checkpoint_manager.sampling_version=step
        trainer.checkpoint_manager.update_weights=update
        def remote(data):
            rollout_calls.append(1)
            for row,info in enumerate(data.non_tensor_batch['shopping']):
                info['sampling_policy_versions']=[trainer.checkpoint_manager.sampling_version]*int(data.batch['response_mask'][row].sum())
            data.meta_info={'metrics':[{}]}
            future=Future();future.future=lambda:future;future.set_result(data)
            return future
        trainer.async_rollout_manager.agent_loop_workers=[SimpleNamespace(generate_sequences=SimpleNamespace(remote=remote))]
        trainer.async_rollout_manager._performance_metrics=lambda metrics,output:{}
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer.use_rm = trainer.use_reference_policy = trainer.use_critic = False
    return SimpleNamespace(trainer=trainer, optimizer=optimizer, scheduler=scheduler, parameter=parameter,
        events=events, weights=weights, sleeps=sleeps, saves=saves, validations=validations,
        observed=observed, calls=rollout_calls, path=diagnostic_path)


def test_native_loop_skips_whole_batch_without_advancing_optimizer_or_save(monkeypatch, tmp_path):
    run = native_harness(monkeypatch, tmp_path, [candidate(True), candidate(), candidate(True), candidate()])
    run.trainer.fit()
    assert len(run.calls) == 4 and len(run.observed) == 2
    assert run.optimizer.state[run.parameter]['step'] == 2 and run.scheduler.last_epoch == 2
    assert run.weights == [0, 1, 2] and run.sleeps == [1, 2]
    assert run.saves == run.validations == [1, 2]
    assert all(len(data) == 4 for data in run.observed)
    for data in run.observed:
        expected, _ = flash.compute_flashreinforce_advantage(data.batch['token_level_rewards'],
            data.batch['response_mask'], data.non_tensor_batch['uid'])
        torch.testing.assert_close(data.batch['advantages'], expected)
    assert not torch.equal(run.parameter.detach(), torch.zeros_like(run.parameter))
    events = [json.loads(line) for line in run.path.read_text().splitlines()]
    updates = [e for e in events if e['event'] == 'optimizer_step']
    admissions = [e for e in events if e['event'] == 'flash_batch_admission']
    assert [e['global_step'] for e in updates] == [1, 2]
    assert [e['decision'] for e in admissions] == ['skip_batch', 'admit', 'skip_batch', 'admit']
    assert [e['global_step'] for e in admissions] == [0, 0, 1, 1]
    assert all(len(e['rollouts']) == 4 for e in admissions)
    assert admissions[0]['invalid_trajectories'] == 1
    sidecar = tmp_path/'checkpoints/global_step_2/flash_batch_admission.json'
    assert json.loads(sidecar.read_text())['candidate_batches'] == 4


@pytest.mark.parametrize('policy,match', [('error', 'policy tokens'), ('skip_batch', 'consecutive')])
def test_strict_policy_and_bounded_skips_stop_without_any_update(monkeypatch, tmp_path, policy, match):
    run = native_harness(monkeypatch, tmp_path, [candidate(True)]*3, policy=policy, consecutive=2)
    with pytest.raises((RuntimeError, ValueError), match=match):
        run.trainer.fit()
    assert not run.optimizer.state and run.scheduler.last_epoch == 0
    assert run.weights == [0] and run.saves == run.validations == []
    assert len(run.calls) == (1 if policy == 'error' else 2)


def test_candidate_budget_stops_before_an_extra_rollout(monkeypatch, tmp_path):
    run = native_harness(monkeypatch, tmp_path, [candidate(True), candidate(), candidate()], target=2, limit=2)
    with pytest.raises(RuntimeError, match='candidate'):
        run.trainer.fit()
    assert len(run.calls) == 2 and len(run.observed) == 1
    assert run.optimizer.state[run.parameter]['step'] == 1


def test_valid_constant_rewards_are_admitted_without_selection(monkeypatch, tmp_path):
    run = native_harness(monkeypatch, tmp_path, [candidate(equal=True)], target=1)
    run.trainer.fit()
    assert len(run.observed) == 1 and run.optimizer.state[run.parameter]['step'] == 1
    assert run.observed[0].batch['advantages'].count_nonzero() == 0


def test_epoch_exhaustion_cannot_report_a_completed_budget(monkeypatch, tmp_path):
    run = native_harness(monkeypatch, tmp_path, [candidate()], target=2)
    with pytest.raises(RuntimeError, match='exhausted'):
        run.trainer.fit()


def test_64_actual_updates_with_32_skips_preserve_checkpoint_and_validation_intervals(monkeypatch, tmp_path):
    candidates = [candidate(invalid=(i % 3 == 0)) for i in range(96)]
    run = native_harness(monkeypatch, tmp_path, candidates, target=64, limit=128)
    run.trainer.config.trainer.save_freq = run.trainer.config.trainer.test_freq = 32
    run.trainer.fit()
    assert len(run.calls) == 96 and len(run.observed) == 64
    assert run.optimizer.state[run.parameter]['step'] == 64 and run.scheduler.last_epoch == 64
    assert run.weights == list(range(65))
    assert run.saves == run.validations == [32, 64]
    events = [json.loads(line) for line in run.path.read_text().splitlines()]
    admissions = [e for e in events if e['event'] == 'flash_batch_admission']
    updates = [e for e in events if e['event'] == 'optimizer_step']
    assert [e['global_step'] for e in updates] == list(range(1, 65))
    assert sum(e['decision'] == 'skip_batch' for e in admissions) == 32
    uids = [rollout['uid'] for e in admissions for rollout in e['rollouts']]
    assert len(uids) == len(set(uids)) == 384
    state = json.loads((tmp_path/'checkpoints/global_step_64/flash_batch_admission.json').read_text())
    assert state['optimizer_updates'] == 64 and state['candidate_batches'] == 96 and state['skipped_batches'] == 32


def test_hook_refuses_source_drift():
    from shopping_grpo.training.grpo.flash_batching import prepare_fit
    with pytest.raises(RuntimeError, match='exact pinned'):
        prepare_fit(patched_source()+'\n', {})


@pytest.mark.parametrize('key,value', [
    ('invalid_batch_policy', 'drop_survivors'), ('max_candidate_batches', 0),
    ('max_candidate_batches', True), ('max_consecutive_skipped_batches', 0),
    ('max_candidate_batches', 1),
])
def test_admission_rejects_unbounded_or_inconsistent_settings(key, value):
    from shopping_grpo.training.grpo.flash_batching import validate_admission_config
    settings = config()
    settings['shopping_flashreinforce'].update(invalid_batch_policy='skip_batch',
        max_candidate_batches=256, max_consecutive_skipped_batches=16)
    settings['shopping_flashreinforce'][key] = value
    settings['trainer'].update(total_training_steps=64, resume_mode='disable')
    with pytest.raises(ValueError):
        validate_admission_config(settings)


def test_admission_refuses_rollout_reuse_including_previously_skipped_uids(tmp_path):
    from shopping_grpo.training.grpo.flash_batching import WholeBatchAdmission
    settings = config()
    settings['shopping_flashreinforce'].update(invalid_batch_policy='skip_batch',
        max_candidate_batches=256, max_consecutive_skipped_batches=16)
    settings['trainer']['resume_mode'] = 'disable'
    admission = WholeBatchAdmission(settings, tmp_path/'diagnostics.jsonl')
    data = candidate(True)
    assert admission.consider(data, 1) is False
    with pytest.raises(ValueError, match='reused'):
        admission.consider(data, 1)


def test_admission_requires_diagnostics_and_rejects_ambiguous_resume(tmp_path):
    from shopping_grpo.training.grpo.flash_batching import WholeBatchAdmission
    settings = config()
    settings['shopping_flashreinforce'].update(invalid_batch_policy='skip_batch',
        max_candidate_batches=256, max_consecutive_skipped_batches=16)
    settings['trainer']['resume_mode'] = 'auto'
    with pytest.raises(ValueError, match='resume_mode=disable'):
        WholeBatchAdmission(settings, tmp_path/'diagnostics.jsonl')
    settings['trainer']['resume_mode'] = 'disable'
    with pytest.raises(ValueError, match='diagnostics'):
        WholeBatchAdmission(settings, '')


def test_hook_leaves_other_methods_and_strict_fit_unchanged():
    from shopping_grpo.training.grpo.flash_batching import install_batch_admission
    calls = []
    class Trainer:
        def fit(self):
            calls.append('native')
    install_batch_admission(Trainer)
    trainer = Trainer()
    for options in ({'enable': False}, {'enable': True, 'invalid_batch_policy': 'error'}):
        trainer.config = {'shopping_flashreinforce': options}
        trainer.fit()
    assert calls == ['native', 'native']
