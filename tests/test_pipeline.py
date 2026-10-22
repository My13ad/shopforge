"""Exercise stage handoffs and refuse an incorrectly pinned new SFT parent."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_preview_has_correct_parent_chain_and_never_launches_a_child(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: pytest.fail('preview started a process'))
    preview = load('pipeline_preview', ROOT / 'tools/preview.py')
    config = json.loads((ROOT / 'configs/paths.example.json').read_text())
    stages = {name: preview.build(name, config) for name in preview.STAGES}
    def value(stage, key):
        args = stages[stage]['argv']; return args[args.index(key) + 1]
    assert value('sft', '--output') == value('sft-reference', '--adapter') == value('opd32', '--adapter')
    assert Path(value('opd32', '--run')) / 'round-32/training/adapter' == Path(value('flashreinforce', '--adapter'))
    assert value('sft-reference', '--output') == value('opd32', '--reference')
    assert value('sft', '--model') == value('opd32', '--student-model') == value('flashreinforce', '--model')
    assert '--dry-run' in stages['flashreinforce']['argv']
    assert all(not r['execution_started'] for r in stages.values())


def test_new_sft_parent_must_match_both_reference_and_explicit_digest(tmp_path):
    common = load('pipeline_common', ROOT / 'opd/runtime/common.py')
    adapter = tmp_path / 'adapter'; adapter.mkdir()
    weights = adapter / 'adapter_model.safetensors'; weights.write_bytes(b'fresh SFT fixture')
    digest = common.sha(weights)
    assert common.verify_initial_adapter(adapter, {'adapter_sha256': digest}, digest) == digest
    with pytest.raises(ValueError): common.verify_initial_adapter(adapter, {'adapter_sha256': '0' * 64}, digest)
    with pytest.raises(ValueError): common.verify_initial_adapter(adapter, {'adapter_sha256': digest}, '0' * 64)
    weights.write_bytes(b'different actual weights')
    with pytest.raises(ValueError): common.verify_initial_adapter(adapter, {'adapter_sha256': digest}, digest)


def test_explicit_model_and_python_paths_replace_historical_root_layout(tmp_path):
    common = load('pipeline_common_paths', ROOT / 'opd/runtime/common.py')
    options = SimpleNamespace(shop_root=tmp_path / 'unused', student_model=tmp_path / 'base', teacher_model=tmp_path / 'teacher',
                              student_python=tmp_path / 'venvs/client/python', teacher_python=tmp_path / 'venvs/server/python')
    paths = common.runtime_paths(options)
    assert paths['model'] == options.student_model and paths['teacher'] == options.teacher_model
    assert paths['python'] == options.student_python and paths['teacher_python'] == options.teacher_python


@pytest.mark.parametrize('ids', [[1, 1], [1, 3], [1, 4], [1, 5], ['1', 2]])
def test_pinned_opd_task_manifest_rejects_duplicates_heldouts_and_calibration(tmp_path, ids):
    common = load('pipeline_common_ids', ROOT / 'opd/runtime/common.py')
    path = tmp_path / 'tasks.jsonl'; path.write_text(''.join(json.dumps({'task_id': t}) + '\n' for t in ids))
    with pytest.raises(ValueError): common.manifest_training_ids(path, [1, 2, 5], [3], [4], [5], 42, 2)


def test_actual_opd32_task_order_is_preserved_in_portable_manifest():
    common = load('pipeline_common_actual_ids', ROOT / 'opd/runtime/common.py')
    train = [r['task_id'] for r in common.rows(ROOT / 'sft/data/sft/train.jsonl')]
    dev = [r['task_id'] for r in common.rows(ROOT / 'sft/data/sft/validation.jsonl')]
    final = [r['task_id'] for r in common.rows(ROOT / 'sft/data/evaluation/tasks.jsonl')]
    cal = common.read(ROOT / 'opd/assets/calibration/run-20261013-1/result/run_config.json')['task_manifest']['task_ids']
    path = ROOT / 'opd/assets/historical-opd32-task-ids.jsonl'
    result = common.manifest_training_ids(path, train, dev, final, cal, 20261014, 256)
    assert result == [r['task_id'] for r in common.rows(path)]


def test_opd_mathematical_and_execution_kernels_match_the_verified_snapshot():
    verify = load('pipeline_verifier', ROOT / 'tools/verify_local.py')
    for name in ('importance_loss', 'extract_teacher_scores', 'choose_training_ids'):
        assert verify.function(ROOT / 'opd/runtime/common.py', name) == verify.function(ROOT / 'provenance/verified-opd32/common.py', name)
    for name in ('rollout', 'score', 'train', 'analyse'):
        assert verify.function(ROOT / 'opd/runtime/phases.py', name) == verify.function(ROOT / 'provenance/verified-opd32/phases.py', name)
