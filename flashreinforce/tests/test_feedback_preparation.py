"""Dry-run real public recipes and fail closed on changed confirmation/gate data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from shopping_grpo.evaluation.frozen_tasks import validate_frozen_evaluation
from shopping_grpo.training.grpo.feedback_sampling import sha256
from test_checkpoint_selection import infos
from test_grpo_entry_and_hook import entry, parent  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'outputs/preparation/flashreinforce-feedback-v1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('stage', ['reference-parent', 'pilot', 'reference-pilot-step1', 'control', 'hard20', 'confirm-parent', 'confirm-control', 'confirm-hard20'])
def test_every_operational_recipe_composes_real_hydra_and_preflight_without_execution(monkeypatch, parent, tmp_path, stage):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from shopping_grpo.training.grpo.flashreinforce import validate_flashreinforce_config
    from shopping_grpo.training.grpo.checkpoint_selection import validate_selection_config
    operations = load('feedback_commands', HERE / 'commands.py')
    model, adapter = parent
    public = operations.command(stage, model=model, adapter=adapter, output_root=tmp_path)
    monkeypatch.setattr(sys, 'argv', public[1:])
    options = entry.parse_args()
    assert options.dry_run
    command, environment = entry.build_command(options)
    for key, value in environment.items(): monkeypatch.setenv(key, value)
    GlobalHydra.instance().clear()
    search = (ROOT / 'outputs/preparation/v1/runtime-wheel/source/verl/trainer/config').as_posix()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
        recipe = compose(config_name='grpo', overrides=command[5:] + [f'hydra.searchpath=["file://{search}"]'])
    checker = load('feedback_preflight', ROOT / 'scripts/check_grpo_runtime.py')
    checker.validate_training_task_isolation(recipe)
    assert Path(recipe.actor_rollout_ref.model.lora_adapter_path) == adapter
    assert Path(recipe.actor_rollout_ref.rollout.agent.agent_loop_config_path).is_file()
    validate_flashreinforce_config(recipe)
    validate_selection_config(recipe)
    if stage in ('control', 'hard20'):
        assert recipe.trainer.total_training_steps == 256 and recipe.trainer.total_epochs == 1
        assert recipe.actor_rollout_ref.actor.optim.lr == 5e-7
    if stage == 'pilot': assert recipe.trainer.total_training_steps == 1
    if stage.startswith('confirm-'):
        assert recipe.trainer.val_only and recipe.shopping_evaluation.task_manifest
        # The same confirmation tasks cannot become training or validation50.
        recipe.shopping_evaluation.enable = False
        recipe.trainer.val_only = False
        recipe.data.train_files = recipe.data.val_files
        recipe.data.val_files = [str(ROOT / 'data/grpo/validation.parquet')]
        recipe.shopping_feedback_sampling.enable = True
        recipe.shopping_feedback_sampling.manifest = str(HERE / 'pool-learnable/sampling-manifest.json')
        recipe.shopping_feedback_sampling.hard_fraction = .2
        with pytest.raises(ValueError): checker.validate_training_task_isolation(recipe)
    monkeypatch.setattr(entry.subprocess, 'call', lambda *a, **kw: pytest.fail('dry-run attempted to execute'))
    entry.main()
    assert not (tmp_path / stage).exists()


def test_confirmation_is_frozen_disjoint_and_keeps_original_public_prompt():
    import pyarrow.parquet as pq
    manifest = validate_frozen_evaluation(HERE / 'confirmation/manifest.json', HERE / 'confirmation/tasks.parquet', root=ROOT)
    rows = pq.read_table(HERE / 'confirmation/tasks.parquet').to_pylist()
    original = pq.read_table(ROOT / 'outputs/preparation/v1/candidate-price953/train953.parquet').to_pylist()[0]
    assert len(rows) == 200 and all(r['prompt'][0] == original['prompt'][0] for r in rows)
    assert all(r['reward_model']['ground_truth'] is None and set(r['extra_info']) == {'split', 'task_id', 'index'} for r in rows)
    audit = json.loads((HERE / 'confirmation/exclusion-audit.json').read_text())
    assert not set(manifest['task_ids']) & set(audit['excluded_task_ids'])
    assert audit['within_set_unique_asins'] == 200 and audit['cross_split_asin_overlap'] == 0


@pytest.mark.parametrize('mutation', ['duplicate', 'final', 'audit', 'parquet'])
def test_confirmation_rejects_changed_ids_or_provenance(tmp_path, mutation):
    target = tmp_path / 'confirmation'
    shutil.copytree(HERE / 'confirmation', target)
    path = target / 'manifest.json'
    data = json.loads(path.read_text())
    if mutation == 'duplicate': data['task_ids'][0] = data['task_ids'][1]
    elif mutation == 'final': data['task_ids'][0] = json.loads((ROOT / 'data/evaluation/tasks.jsonl').read_text().splitlines()[0])['task_id']
    elif mutation == 'audit': (target / 'exclusion-audit.json').write_text('{}')
    else: (target / 'tasks.parquet').write_bytes(b'changed')
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): validate_frozen_evaluation(path, target / 'tasks.parquet', root=ROOT)


@pytest.mark.parametrize('mutation', [None, 'behavior', 'missing', 'stale', 'readback'])
def test_gpu_acceptance_requires_both_parent_and_post_update_agreement(tmp_path, mutation):
    gate = load('feedback_gpu_gate', HERE / 'check_gpu_acceptance.py')
    pilot, parent_ref, step_ref, adapter = [tmp_path / n for n in ('pilot', 'parent', 'step', 'adapter')]
    for p in (pilot, parent_ref, step_ref, adapter): p.mkdir()
    (adapter / 'adapter_model.safetensors').write_bytes(b'changed step1 weights; CPU test fixture only')
    step_sha = sha256(adapter / 'adapter_model.safetensors')
    pilot_launch = {'parent_adapter_sha256': gate.PARENT_SHA, 'method': 'flashreinforce', 'checkpoint_selection': {'interval': 1},
                    'environment_manifest_sha256': 'fixture', 'command': ['trainer.total_training_steps=1', 'shopping_feedback_sampling.hard_fraction=0.0', 'shopping_flashreinforce.execution=async']}
    (pilot / 'launch.json').write_text(json.dumps(pilot_launch))
    (pilot / 'training_diagnostics.jsonl').write_text(json.dumps({'event': 'optimizer_step', 'global_step': 1}))
    (pilot / 'checkpoint_selection').mkdir()
    ids = [json.loads(s)['task_id'] for s in (ROOT / 'data/grpo/validation.jsonl').read_text().splitlines() if s]
    for step, reference, digest in [(0, parent_ref, gate.PARENT_SHA), (1, step_ref, step_sha)]:
        records = infos(30 + step, step)
        for row, task in zip(records, ids):
            row.update(task_id=task, actions=[{'tool': 'search_products', 'parameters': {'query': 'fixture'}}], guard_rejections=0, action_attempts=5)
        native = {'task_records': deepcopy(records), 'validation_parquet_sha256': sha256(ROOT / 'data/grpo/validation.parquet')}
        if mutation == 'stale' and step == 1: native['task_records'][0]['sampling_policy_versions'] = [0, 0]
        (pilot / f'checkpoint_selection/validation-step-{step}.json').write_text(json.dumps(native))
        (reference / 'task_rollouts').mkdir(); (reference / 'runtime_audit').mkdir()
        command = ['trainer.val_only=true', 'data.val_batch_size=1', 'actor_rollout_ref.rollout.agent.num_workers=1',
                   'actor_rollout_ref.rollout.enforce_eager=true', 'actor_rollout_ref.rollout.enable_prefix_caching=false',
                   'actor_rollout_ref.rollout.val_kwargs.n=1', 'actor_rollout_ref.rollout.val_kwargs.do_sample=false']
        launch = {'profile': 'eval', 'parent_adapter_sha256': digest, 'training_reward_overlay': 'native', 'command': command, 'environment_manifest_sha256': 'fixture'}
        (reference / 'launch.json').write_text(json.dumps(launch))
        for i, row in enumerate(records):
            if mutation == 'behavior' and step == 1 and i == 0: row['actions'] = []
            if mutation == 'missing' and step == 0 and i == 0: continue
            (reference / f'task_rollouts/{i}.json').write_text(json.dumps({'receipt_scope': 'post_native_unpad', 'adapter_sha256': digest, 'shopping': row}))
        samples = [{'name_suffix': str(i), 'shape': [1], 'values_bf16': [1]} for i in range(24)]
        audits = [{'kind': 'actor_adapter', 'all_tensors_match_selected_file_in_live_dtype': True, 'tensor_count': 372},
                  {'kind': 'sender', 'samples': samples}, {'kind': 'receiver', 'samples': samples, 'readback_matches': not (mutation == 'readback' and step == 1)}]
        audits += [{'kind': 'identity', 'training_task_id': t, 'repeats': [{'prompt_logprobs': [1], 'log_probs': [1]}] * 2} for t in (53, 147)]
        for i, r in enumerate(audits):
            (reference / f'runtime_audit/{i}.json').write_text(json.dumps(dict(r, adapter_sha256=digest)))
    if mutation is None:
        assert gate.compare(pilot, parent_ref, step_ref, adapter)['passed']
    else:
        with pytest.raises(ValueError): gate.compare(pilot, parent_ref, step_ref, adapter)
