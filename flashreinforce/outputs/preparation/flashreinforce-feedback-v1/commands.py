"""Print public-entry argv templates. This module never launches any process."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT_SHA256 = '466b8f402a21d0cd7ad63d97ceaf682dd7a8699a6c084c3dcc2754fdb8ad4fe1'
STAGES = ('reference-parent', 'pilot', 'reference-pilot-step1', 'control', 'hard20',
          'confirm-parent', 'confirm-control', 'confirm-hard20')


def command(stage, *, model, adapter, output_root, env_url='http://127.0.0.1:5721', dry_run=True):
    if stage not in STAGES:
        raise ValueError('unknown stage')
    result = [sys.executable, str(ROOT / 'scripts/train_grpo.py'), '--model', str(model), '--adapter', str(adapter),
              '--seed', '42', '--train-data', str(ROOT / 'outputs/preparation/v1/candidate-price953/train953.parquet'),
              '--env-url', env_url, '--output', str(Path(output_root) / stage), '--experiment-name', 'feedback-v1-' + stage]
    if stage in ('pilot', 'control', 'hard20'):
        pilot = stage == 'pilot'
        result += ['--method', 'flashreinforce', '--profile', 'main', '--flash-execution', 'async',
                   '--flash-invalid-batch-policy', 'skip_batch', '--flash-batch-size', '8', '--checkpoint-selection',
                   '--validation-interval', '1' if pilot else '64', '--early-stopping-minimum-updates', '0' if pilot else '128',
                   '--early-stopping-patience', '1' if pilot else '3',
                   '--feedback-manifest', str(HERE / 'pool-learnable/sampling-manifest.json'),
                   '--hard-task-fraction', '0.2' if stage == 'hard20' else '0.0']
        extras = [f'trainer.total_training_steps={1 if pilot else 256}', 'trainer.total_epochs=1',
                  'actor_rollout_ref.actor.optim.lr=5e-7', 'actor_rollout_ref.actor.optim.lr_warmup_steps=1',
                  'actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=true',
                  f'shopping_flashreinforce.max_candidate_batches={32 if pilot else 1024}']
    else:
        confirmation = stage.startswith('confirm-')
        result += ['--method', 'grpo', '--profile', 'eval', '--val-data',
                   str(HERE / 'confirmation/tasks.parquet' if confirmation else ROOT / 'data/grpo/validation.parquet'),
                   '--identity-contexts', str(HERE / ('identity-contexts-confirmation.json' if confirmation else 'identity-contexts-validation.json'))]
        if confirmation:
            result += ['--evaluation-task-manifest', str(HERE / 'confirmation/manifest.json')]
        extras = []
    if dry_run:
        result += ['--dry-run']
    return result + ['--', 'ray_kwargs.ray_init.num_cpus=20', *extras]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--env-url', default='http://127.0.0.1:5721')
    args = parser.parse_args()
    print(json.dumps({'stage': args.stage, 'execution_started': False,
                      'argv': command(args.stage, model=args.model, adapter=args.adapter, output_root=args.output_root, env_url=args.env_url)}, indent=2))


if __name__ == '__main__':
    main()
