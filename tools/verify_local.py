"""Verify source syntax, frozen data, parent isolation and unchanged OPD math on CPU."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '__pycache__', '.pytest_cache', '.venv', '.venv-shopsim', '.venvs', 'checks', 'runs'}


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def static_files(root=ROOT):
    return sorted(p for p in root.rglob('*') if p.is_file() and not any(part in SKIP for part in p.relative_to(root).parts)
                  and p.relative_to(root).parts[0] != 'models'
                  and p.name not in {'items_eval_train.json', 'products.sqlite3'}
                  and p.suffix not in {'.pyc', '.pyo'} and p.name not in {'file-manifest.json', 'verification-report.json'})


def task_ids(path):
    return [int(json.loads(line)['task_id']) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def function(path, name):
    node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    return ast.dump(node, include_attributes=False)


def verify(*, check_origins=False):
    files = static_files()
    pyfiles = [p for p in files if p.suffix == '.py']
    for path in pyfiles: ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    for path in files:
        if path.name == '.env' or path.suffix in {'.safetensors', '.pt', '.pth', '.bin', '.sqlite3'}:
            raise ValueError('unexpected credential/model/generated artifact: ' + str(path.relative_to(ROOT)))
        if path.suffix in {'.py', '.json', '.jsonl', '.md', '.txt', '.toml', '.yaml', '.yml', '.sh'}:
            value = path.read_text(encoding='utf-8-sig')
            if re.search(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}|\bgh[pousr]_[A-Za-z0-9]{24,}', value):
                raise ValueError('possible embedded secret; inspect locally: ' + str(path.relative_to(ROOT)))
    sft_train, sft_dev = [set(task_ids(ROOT / 'sft/data/sft' / name)) for name in ('train.jsonl', 'validation.jsonl')]
    final = set(task_ids(ROOT / 'sft/data/evaluation/tasks.jsonl'))
    rl_train = set(task_ids(ROOT / 'flashreinforce/outputs/preparation/v1/candidate-price953/train953.jsonl')) if (ROOT / 'flashreinforce/outputs/preparation/v1/candidate-price953/train953.jsonl').exists() else set(json.loads((ROOT / 'flashreinforce/outputs/preparation/flashreinforce-feedback-v1/pool-learnable/sampling-manifest.json').read_text(encoding='utf-8'))['train_task_ids'])
    rl_val = set(task_ids(ROOT / 'flashreinforce/data/grpo/validation.jsonl'))
    opd = task_ids(ROOT / 'opd/assets/historical-opd32-task-ids.jsonl')
    cal = json.loads((ROOT / 'opd/assets/calibration/run-20261013-1/result/run_config.json').read_text(encoding='utf-8'))
    calibration = set(cal['task_manifest']['task_ids'])
    assert (len(sft_train), len(sft_dev), len(final), len(rl_train), len(rl_val), len(opd)) == (800, 200, 200, 953, 50, 256)
    assert len(set(opd)) == 256 and set(opd) <= sft_train and not set(opd) & calibration
    assert not sft_train & sft_dev and not (sft_train | sft_dev | rl_train | rl_val | set(opd)) & final
    assert not rl_train & (sft_train | sft_dev | rl_val)
    historical = ROOT / 'provenance/verified-opd32'
    active = ROOT / 'opd/runtime'
    for name in ('importance_loss', 'extract_teacher_scores', 'choose_training_ids'):
        assert function(active / 'common.py', name) == function(historical / 'common.py', name), name
    for name in ('rollout', 'score', 'train', 'analyse'):
        assert function(active / 'phases.py', name) == function(historical / 'phases.py', name), name
    engine = 'environments/ShopSimulator/shop_env/web_agent_site/engine/engine.py'
    assert sha(ROOT / 'sft' / engine) == '130e0261c5c589ac133f1562f6d718736cda44abe54e2151867f5b17f136a086'
    assert sha(ROOT / 'flashreinforce' / engine) == '80f0fb1a3fc1bfa3ff26a3eb8aa5a7978a60f75fa4fd79bb1a7cefdd15e2c2ab'
    archive_sha = 'f51c33217061479f9c95a1068621fcd38e4883ae3d2f6a1627037bea934f2125'
    for stage in ('sft', 'flashreinforce'):
        assert sha(ROOT / stage / 'environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz') == archive_sha
    if check_origins:
        for entry in json.loads((ROOT / 'provenance/source-files.json').read_text(encoding='utf-8'))['copies']:
            source = ROOT.parent / entry['source']
            if source.is_file() and not source.resolve().is_relative_to(ROOT):
                assert sha(source) == entry['source_sha256'], str(source)
    return {'verified': True, 'python_files_parsed': len(pyfiles), 'files': len(files),
            'total_bytes': sum(p.stat().st_size for p in files), 'sft_train': 800, 'sft_development': 200,
            'opd_training': 256, 'flash_training': 953, 'flash_validation': 50,
            'original_final': 200, 'opd_loss_rollout_score_train_unchanged': True,
            'copied_product_archives_verified': True, 'secret_pattern_scan_passed': True,
            'origin_check_requested': check_origins, 'performs_gpu_work': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-origins', action='store_true')
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    result = verify(check_origins=args.check_origins)
    if args.write_manifest:
        (ROOT / 'provenance/file-manifest.json').write_text(json.dumps({p.relative_to(ROOT).as_posix(): sha(p) for p in static_files()}, indent=2) + '\n', encoding='utf-8')
        (ROOT / 'provenance/verification-report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
