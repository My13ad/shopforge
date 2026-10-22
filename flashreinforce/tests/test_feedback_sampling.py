"""Real completion scheduling and native CPU optimization for train-only feedback."""
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from shopping_grpo.training.grpo.async_sampling import CompletionPool, Submitted
from shopping_grpo.training.grpo.feedback_sampling import ReservationSampler, load_manifest
from test_flash_batch_admission import candidate, native_harness, Proto
from test_grpo_entry_and_hook import args, entry, parent  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "outputs/preparation/flashreinforce-feedback-v1/pool-learnable/sampling-manifest.json"


def test_out_of_order_workers_preserve_distinct_batches_and_exact_injection_quota():
    # A tiny hard pool forces temporary reservation exhaustion while a slow task
    # and completed tasks coexist. Release happens only after consuming a batch.
    sampler = ReservationSampler(range(30), range(4), hard_fraction=.2, seed=42, limit=200)
    class Source:
        def try_next(self): return sampler.draw()
    executor = ThreadPoolExecutor(4)
    def work(choice):
        Event().wait(.003 if choice['task_id'] % 3 else .015)
        return choice
    def submit(choice):
        future = executor.submit(work, choice)
        return Submitted(future, future.cancel)
    pool = CompletionPool(Source(), submit, inflight=8, capacity=8, timeout=3)
    received = []
    try:
        for _ in range(50):
            batch = [next(pool)[1] for _ in range(4)]
            assert len({c['task_id'] for c in batch}) == 4
            received.extend(batch)
            sampler.release([c['task_id'] for c in batch])
        with pytest.raises(StopIteration): next(pool)
    finally:
        pool.close()
        executor.shutdown(wait=True)
    assert not sampler.reserved
    assert sorted(c['dispatch_index'] for c in received) == list(range(1, 201))
    assert sum(c['lane'] == 'hard' for c in received) == 40
    assert any(c['hard_member'] and c['lane'] == 'background' for c in received)


def test_sampler_draws_repeat_across_batches_but_not_while_reserved_and_is_reproducible():
    def run():
        sampler = ReservationSampler(range(20), range(5), hard_fraction=.2, seed=42, limit=100)
        choices = []
        for _ in range(25):
            batch = [sampler.draw() for _ in range(4)]
            assert len({c['task_id'] for c in batch}) == 4
            choices.extend(batch)
            sampler.release([c['task_id'] for c in batch])
        return choices
    assert run() == run()
    assert len({c['task_id'] for c in run()}) < 100


@pytest.mark.parametrize('mutation', ['final', 'validation', 'duplicate', 'answer', 'fingerprint'])
def test_manifest_rejects_leakage_answers_and_changed_sources(tmp_path, mutation):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shopping_grpo.training.grpo.feedback_sampling import sha256
    data = json.loads(POOL.read_text())
    train = ROOT / 'outputs/preparation/v1/candidate-price953/train953.parquet'
    if mutation in ('final', 'validation'):
        name = 'data/evaluation/tasks.jsonl' if mutation == 'final' else 'data/grpo/validation.jsonl'
        task = json.loads((ROOT / name).read_text().splitlines()[0])['task_id']
        rows = pq.read_table(train).to_pylist()
        rows[0]['extra_info']['task_id'] = task
        train = tmp_path / 'changed.parquet'
        pq.write_table(pa.Table.from_pylist(rows), train)
        data['train_task_ids'][0] = task
        data['train_sha256'] = sha256(train)
    elif mutation == 'duplicate': data['hard_task_ids'].append(data['hard_task_ids'][0])
    elif mutation == 'answer': data['target_asin'] = 'secret'
    else: data['train_sha256'] = '0' * 64
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): load_manifest(path, train, root=ROOT)


def test_miner_uses_all_valid_training_failures_including_whole_skipped_batches(tmp_path):
    spec = importlib.util.spec_from_file_location('feedback_miner', ROOT / 'scripts/prepare_feedback_pool.py')
    miner = importlib.util.module_from_spec(spec); spec.loader.exec_module(miner)
    rows = []
    for i in range(5):
        data = candidate()
        for info in data.non_tensor_batch['shopping']:
            info.update(uid=f'{i}-{info["task_id"]}', reward_valid=True, reward_detail={'reward_valid': True},
                        done=True, terminal_done=True, terminal_over=True,
                        termination_reason='repeat_loop', reward_type='repeat_loop')
            if i == 4:
                info.update(termination_reason='gold_purchase', reward_type='gold_purchase')
        data.non_tensor_batch['shopping'][0]['reward']['sampling_invalid'] = True
        rows.append({'event': 'flash_batch_admission', 'decision': 'skip_batch', 'rollouts': data.non_tensor_batch['shopping']})
    path = tmp_path / 'receipts.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    hard, audit = miner.mine(range(4), [path], size=2)
    assert hard == [1, 2] and audit['tasks'][0]['invalid_diagnostic_only'] == 5
    assert audit['tasks'][1]['valid'] == 5 and audit['tasks'][1]['failure'] == 4 and audit['tasks'][1]['success'] == 1
    failures_only = tmp_path / 'never-successful.jsonl'
    failures_only.write_text('\n'.join(json.dumps(r) for r in rows[:4]))
    with pytest.raises(ValueError, match='not enough'):
        miner.mine(range(4), [failures_only], size=1)
    with pytest.raises(ValueError, match='outside'): miner.mine(range(3), [path], size=1)
    with pytest.raises(ValueError, match='duplicate'): miner.mine(range(4), [path, path], size=2)


@pytest.mark.parametrize('fraction', [0., .2])
@pytest.mark.parametrize('with_selection', [False, True])
def test_real_native_fit_uses_feedback_dataset_and_preserves_native_whole_batch_admission(monkeypatch, tmp_path, fraction, with_selection):
    run = native_harness(monkeypatch, tmp_path, [candidate()] * 30, target=6, execution='async')
    trainer = run.trainer
    manifest = json.loads(POOL.read_text())
    task_ids = manifest['train_task_ids']
    trainer.config.shopping_feedback_sampling = {'enable': True, 'manifest': str(POOL), 'hard_fraction': fraction}
    trainer.config.data.train_files = [str(ROOT / 'outputs/preparation/v1/candidate-price953/train953.parquet')]
    trainer.config.data.seed = 42
    trainer.config.shopping_flashreinforce.max_candidate_batches = 32
    original = candidate()
    class Dataset:
        dataframe = {'extra_info': [{'task_id': task} for task in task_ids]}
        def __getitem__(self, index):
            data = deepcopy(original)
            for info in data.non_tensor_batch['shopping']: info['task_id'] = task_ids[index]
            return Proto(data).select_idxs([index % 4])
    class Loader:
        def __len__(self): return 100
        def __iter__(self): raise AssertionError('feedback must use native dataset rather than a duplicated dataloader')
        @staticmethod
        def collate_fn(rows):
            assert len(rows) == 1
            return rows[0]
    trainer.train_dataset, trainer.train_dataloader = Dataset(), Loader()
    expected_updates = 6
    if with_selection:
        from shopping_grpo.training.grpo.checkpoint_selection import TrainingCheckpointSelection
        from shopping_grpo.training.grpo.diagnostics import persist_unpadded_validation_batch
        from test_checkpoint_selection import infos
        import pyarrow.parquet as pq
        expected_updates = 3
        c = trainer.config
        c.shopping_checkpoint_selection = {'enable': True, 'interval': 1, 'patience': 2, 'minimum_updates': 0}
        c.data.val_files = [str(ROOT / 'data/grpo/validation.parquet')]
        c.data.val_batch_size, c.data.validation_shuffle = 1, False
        c.trainer.val_before_train = True
        c.actor_rollout_ref.rollout.update(enforce_eager=True, enable_prefix_caching=False,
            val_kwargs={'n': 1, 'do_sample': False, 'temperature': 0., 'top_p': 1.})
        c.actor_rollout_ref.rollout.agent.num_workers = 1
        expected_ids = [r['extra_info']['task_id'] for r in pq.read_table(c.data.val_files[0]).to_pylist()]
        monkeypatch.setenv('GRPO_OUTPUT_DIR', str(tmp_path / 'output'))
        monkeypatch.setenv('SHOPPING_GRPO_ROOT', str(ROOT))
        def validate():
            pool = getattr(trainer, '_shopping_flash_pool', None)
            if pool is not None: assert pool.paused.is_set()
            records = infos({0: 30, 1: 31, 2: 31, 3: 30}[trainer.global_steps], trainer.global_steps)
            for record, task in zip(records, expected_ids): record['task_id'] = task
            class ValidationBatch:
                non_tensor_batch = {'shopping': records}
                def __len__(self): return len(records)
            data = ValidationBatch()
            persist_unpadded_validation_batch(data, global_step=trainer.global_steps)
            run.validations.append(trainer.global_steps)
            return {'validation': 1.}
        trainer._validate = validate
        monkeypatch.setattr(TrainingCheckpointSelection, 'pin_best', lambda self, step: {'step': step})
    # A native invalid rollout forces an entire candidate batch to skip, without
    # changing the valid companions' rewards or reusing their UID.
    worker = trainer.async_rollout_manager.agent_loop_workers[0]
    original_remote = worker.generate_sequences.remote
    calls = []
    def remote(data):
        assert all('feedback_sampling' not in r for r in data.non_tensor_batch['shopping'])
        calls.append(1)
        if len(calls) == 1:
            data.non_tensor_batch['shopping'][0]['reward']['sampling_invalid'] = True
        if not with_selection and len(calls) % 17 == 0:
            pending = Future()
            pending.future = lambda: pending
            return pending  # Slow unused tail must be audited as cancellation, not a worker failure.
        return original_remote(data)
    monkeypatch.setattr(worker.generate_sequences, 'remote', remote)
    trainer.fit()
    assert run.optimizer.state[run.parameter]['step'] == expected_updates and run.scheduler.last_epoch == expected_updates
    events = [json.loads(s) for s in run.path.read_text().splitlines()]
    admissions = [r for r in events if r['event'] == 'flash_batch_admission']
    assert admissions[0]['decision'] == 'skip_batch' and len(admissions) == expected_updates + 1
    assert admissions[-1]['feedback_sampling_counts']['admitted']['total'] == 4 * expected_updates
    assert admissions[-1]['feedback_sampling_counts']['skipped']['total'] == 4
    assert all(len({r['task_id'] for r in e['rollouts']}) == 4 for e in admissions)
    for e in admissions:
        for row in e['rollouts']:
            assert row['reward']['total'] in [1., .55, -.15, -.85]
    audit = Path(trainer.config.trainer.default_local_dir) / 'feedback_sampling.jsonl'
    records = [json.loads(s) for s in audit.read_text().splitlines()]
    dispatch = [r['choice'] for r in records if r['stage'] == 'dispatch']
    assert sum(r['lane'] == 'hard' for r in dispatch) == int(len(dispatch) * fraction)
    assert trainer._shopping_feedback_source.closed and trainer._shopping_flash_pool.closed
    import shutil
    shutil.copyfile(run.path, audit.parent / 'training_diagnostics.jsonl')
    spec = importlib.util.spec_from_file_location('feedback_summary', ROOT / 'scripts/summarize_feedback_sampling.py')
    summary_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(summary_module)
    result = summary_module.summarize(audit.parent)
    assert result['verified'] and result['optimizer_updates'] == expected_updates
    assert result['stages']['admitted']['total'] == result['stages']['optimizer_used']['total'] == 4 * expected_updates
    if with_selection:
        assert run.validations == [0, 1, 2, 3]
        best = json.loads((tmp_path / 'output/checkpoint_selection/best-checkpoint.json').read_text())
        assert best['step'] == 1


def test_public_entry_and_real_hydra_differ_only_in_injection_fraction(monkeypatch, parent, tmp_path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from shopping_grpo.training.grpo.flashreinforce import validate_flashreinforce_config
    options = args(monkeypatch, parent, tmp_path, method='flashreinforce', profile='main', flash_execution='async')
    options.feedback_manifest = POOL
    options.checkpoint_selection = True
    options.hard_task_fraction = 0.
    first, environment = entry.build_command(options)
    options.hard_task_fraction = .2
    second, _ = entry.build_command(options)
    assert [(a, b) for a, b in zip(first, second) if a != b] == [('shopping_feedback_sampling.hard_fraction=0.0', 'shopping_feedback_sampling.hard_fraction=0.2')]
    for key, value in environment.items(): monkeypatch.setenv(key, value)
    GlobalHydra.instance().clear()
    search = (ROOT / 'outputs/preparation/v1/runtime-wheel/source/verl/trainer/config').as_posix()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
        recipe = compose(config_name='grpo', overrides=second[5:] + [f'hydra.searchpath=["file://{search}"]'])
    validate_flashreinforce_config(recipe)
    assert recipe.shopping_feedback_sampling.hard_fraction == .2
    options.hydra_overrides = ['shopping_feedback_sampling.hard_fraction=0.9']
    with pytest.raises(SystemExit, match='dedicated'): entry.build_command(options)
