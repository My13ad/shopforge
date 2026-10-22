"""Print stage commands only. No execution, model loading or service calls."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('sft', 'sft-reference', 'opd32', 'flashreinforce', 'flash-feedback-control', 'flash-feedback-hard20')


def build(stage, config, *, root=ROOT):
    root = Path(root).resolve()
    def path(value):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else root / p)
    base, teacher = path(config['models']['base']), path(config['models']['teacher'])
    sft_python, teacher_python, flash_python = [path(config['python'][name]) for name in ('sft', 'teacher', 'flash')]
    sft = root / 'sft'
    flash = root / 'flashreinforce'
    runs, sft_adapter, reference, opd_adapter = [path(config[name]) for name in ('runs', 'sft_adapter', 'sft_reference', 'opd_adapter')]
    unresolved = []
    if stage == 'sft':
        command = [sft_python, str(sft / 'scripts/train_lora_sft.py'), '--model', base,
            '--train', str(sft / 'data/sft/train.jsonl'), '--validation', str(sft / 'data/sft/validation.jsonl'),
            '--output', sft_adapter, '--max-length', '28672', '--epochs', '3', '--dtype', 'bf16',
            '--per-device-train-batch-size', '1', '--per-device-eval-batch-size', '1',
            '--gradient-accumulation-steps', '8', '--learning-rate', '1e-4', '--warmup-ratio', '0.03',
            '--lora-r', '16', '--lora-alpha', '32', '--lora-dropout', '0.05', '--gradient-checkpointing',
            '--attention-implementation', 'sdpa', '--liger-kernel', '--seed', '42',
            '--loss-mode', 'smartad', '--linear-attention-backend', 'torch']
        cwd, pythonpath = sft, sft / 'src'
    elif stage == 'sft-reference':
        command = [sft_python, str(sft / 'recipes/round_task_eval.py'), '--workspace', str(sft),
            '--model', base, '--adapter', sft_adapter, '--reference-evaluation', str(root / 'opd/assets/reference-e1'),
            '--output', reference, '--count', '200', '--base-url', config['shop_urls']['sft']]
        cwd, pythonpath = sft, sft / 'src'
    elif stage == 'opd32':
        weights = Path(sft_adapter) / 'adapter_model.safetensors'
        if weights.is_file():
            with weights.open('rb') as handle: parent_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
        else:
            parent_sha = '<SFT_ADAPTER_SHA256_AFTER_TRAINING>'
            unresolved.append('SFT adapter and matching fresh reference evaluation are not prepared yet')
        command = [sft_python, str(root / 'opd/runtime/worker.py'), '--workspace', str(sft), '--shop-root', str(root / 'models'),
            '--student-model', base, '--teacher-model', teacher, '--student-python', sft_python,
            '--teacher-python', teacher_python, '--adapter', sft_adapter, '--reference', reference,
            '--calibration', str(root / 'opd/assets/calibration'), '--expected-parent-sha256', parent_sha,
            '--training-task-manifest', str(root / 'opd/assets/historical-opd32-task-ids.jsonl'),
            '--run', str(Path(runs) / 'opd32'), '--training-tasks', '256', '--tasks-per-update', '8',
            '--sampling-seed', '42', '--selection-seed', '20261014', '--timeout-minutes', '720',
            '--keep-optimizer-every', '0', '--min-free-gib', '4', '--shop-url', config['shop_urls']['sft']]
        cwd, pythonpath = sft, sft / 'src'
    elif stage == 'flashreinforce':
        command = [flash_python, str(flash / 'scripts/train_grpo.py'), '--model', base, '--adapter', opd_adapter,
            '--method', 'flashreinforce', '--flash-execution', 'async', '--profile', 'main',
            '--checkpoint-selection', '--train-data', str(flash / 'outputs/preparation/v1/candidate-price953/train953.parquet'),
            '--env-url', config['shop_urls']['flash'], '--output', str(Path(runs) / 'flash-from-opd32'),
            '--seed', '42', '--dry-run', '--', 'trainer.total_training_steps=256',
            'shopping_flashreinforce.max_candidate_batches=1024']
        cwd, pythonpath = flash, flash / 'src'
    elif stage in ('flash-feedback-control', 'flash-feedback-hard20'):
        if not config.get('flash912_adapter'):
            unresolved.append('flash912_adapter is required for the separate retrospective feedback experiment')
        adapter = path(config.get('flash912_adapter', 'models/retained-flash912-adapter'))
        command = [flash_python, str(flash / 'outputs/preparation/flashreinforce-feedback-v1/commands.py'),
            '--stage', 'control' if stage.endswith('control') else 'hard20', '--model', base,
            '--adapter', adapter, '--output-root', str(Path(runs) / 'flash-feedback'), '--env-url', config['shop_urls']['flash']]
        cwd, pythonpath = flash, flash / 'src'
    else:
        raise ValueError('unknown stage')
    if not (Path(base) / 'config.json').is_file():
        unresolved.append('base model files must be supplied at the configured path')
    if not Path(command[0]).is_file():
        unresolved.append('the configured stage Python environment has not been supplied')
    if stage in ('sft-reference', 'opd32') and not (Path(sft_adapter) / 'adapter_model.safetensors').is_file():
        unresolved.append('the SFT stage must produce its adapter first')
    if stage == 'opd32':
        if not (Path(reference) / 'run_config.json').is_file():
            unresolved.append('the fresh SFT reference evaluation must be completed first')
        if not (Path(teacher) / 'config.json').is_file():
            unresolved.append('the configured FP8 teacher model files must be supplied')
    if stage == 'flashreinforce' and not (Path(opd_adapter) / 'adapter_model.safetensors').is_file():
        unresolved.append('the OPD stage must produce its final adapter first')
    return {'stage': stage, 'cwd': str(cwd), 'env': {'PYTHONPATH': str(pythonpath)}, 'argv': command,
            'unresolved_inputs': unresolved, 'execution_started': False,
            'note': 'This is a command preview. Training and evaluation are separate explicit actions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/paths.example.json')
    args = parser.parse_args()
    print(json.dumps(build(args.stage, json.loads(args.config.read_text(encoding='utf-8'))), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
