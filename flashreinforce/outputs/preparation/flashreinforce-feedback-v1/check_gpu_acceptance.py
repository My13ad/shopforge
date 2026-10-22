"""Compare parent and post-refit validation to independent audited engine loads."""
import argparse
import json
from pathlib import Path

from shopping_grpo.training.grpo.diagnostics import samples_equal, validate_task_records
from shopping_grpo.training.grpo.checkpoint_selection import summarize_validation
from shopping_grpo.training.grpo.feedback_sampling import sha256

PARENT_SHA = '466b8f402a21d0cd7ad63d97ceaf682dd7a8699a6c084c3dcc2754fdb8ad4fe1'
ROOT = Path(__file__).resolve().parents[3]


def independent(run, expected_ids, expected_sha):
    launch = json.loads((run / 'launch.json').read_text())
    if launch['parent_adapter_sha256'] != expected_sha or launch['profile'] != 'eval' or launch['training_reward_overlay'] != 'native':
        raise ValueError('independent reference launch differs')
    required = {'trainer.val_only=true', 'data.val_batch_size=1', 'actor_rollout_ref.rollout.agent.num_workers=1',
                'actor_rollout_ref.rollout.enforce_eager=true', 'actor_rollout_ref.rollout.enable_prefix_caching=false',
                'actor_rollout_ref.rollout.val_kwargs.n=1', 'actor_rollout_ref.rollout.val_kwargs.do_sample=false'}
    if not required <= set(launch['command']):
        raise ValueError('independent reference protocol differs')
    records = [json.loads(p.read_text()) for p in (run / 'task_rollouts').glob('*.json')]
    tasks = validate_task_records(records, set(expected_ids), expected_sha)
    if len(records) != 50:
        raise ValueError('independent reference denominator differs')
    for row in tasks.values():
        overlay = row['training_reward_overlay']
        if overlay['version'] != 'native' or overlay['applied'] or overlay['delta'] != 0:
            raise ValueError('independent reference reward overlay differs')
    audits = [json.loads(p.read_text()) for p in (run / 'runtime_audit').glob('*.json')]
    if not audits or any(r['adapter_sha256'] != expected_sha for r in audits):
        raise ValueError('missing or mixed independent live-weight audit')
    kinds = {kind: [r for r in audits if r['kind'] == kind] for kind in ('actor_adapter', 'sender', 'receiver', 'identity')}
    if not all(kinds.values()) or len(kinds['identity']) != 2:
        raise ValueError('incomplete independent weight audit')
    if {r['training_task_id'] for r in kinds['identity']} != {53, 147} or not all(len(r['repeats']) == 2 and all(x['prompt_logprobs'] and x['log_probs'] for x in r['repeats']) for r in kinds['identity']):
        raise ValueError('independent identity contexts differ')
    if not all(r['all_tensors_match_selected_file_in_live_dtype'] and r['tensor_count'] == 372 for r in kinds['actor_adapter']):
        raise ValueError('live adapter mismatch')
    left = {v['name_suffix']: v for r in kinds['sender'] for v in r['samples']}
    right = {v['name_suffix']: v for r in kinds['receiver'] for v in r['samples']}
    if len(left) != 24 or set(left) != set(right) or not all(samples_equal(v, right[k]) for k, v in left.items()) or not all(r['readback_matches'] for r in kinds['receiver']):
        raise ValueError('refit readback mismatch')
    return tasks, launch


def compare(pilot, parent_reference, step1_reference, step1_adapter):
    expected = [json.loads(s)['task_id'] for s in (ROOT / 'data/grpo/validation.jsonl').read_text().splitlines() if s]
    pilot_launch = json.loads((pilot / 'launch.json').read_text())
    if pilot_launch['parent_adapter_sha256'] != PARENT_SHA or pilot_launch['method'] != 'flashreinforce' or not pilot_launch.get('checkpoint_selection'):
        raise ValueError('pilot must start from the retained912 adapter with checkpoint selection')
    if not {'trainer.total_training_steps=1', 'shopping_feedback_sampling.hard_fraction=0.0', 'shopping_flashreinforce.execution=async'} <= set(pilot_launch['command']):
        raise ValueError('pilot recipe differs')
    diagnostics = [json.loads(s) for s in (pilot / 'training_diagnostics.jsonl').read_text().splitlines() if s]
    if [r['global_step'] for r in diagnostics if r['event'] == 'optimizer_step'] != [1]:
        raise ValueError('pilot must contain exactly one committed optimizer update')
    step1_sha = sha256(step1_adapter / 'adapter_model.safetensors')
    if step1_sha == PARENT_SHA:
        raise ValueError('pilot weights did not change')
    comparisons = []
    for step, reference, digest in [(0, parent_reference, PARENT_SHA), (1, step1_reference, step1_sha)]:
        records = json.loads((pilot / f'checkpoint_selection/validation-step-{step}.json').read_text())
        infos = records['task_records']
        summarize_validation(infos, expected, step=step, asynchronous=True)
        tasks, launch = independent(reference, expected, digest)
        if pilot_launch['environment_manifest_sha256'] != launch['environment_manifest_sha256']:
            raise ValueError('pilot and reference environment differ')
        if records['validation_parquet_sha256'] != sha256(ROOT / 'data/grpo/validation.parquet'):
            raise ValueError('pilot validation data differs')
        fields = ('task_id', 'actions', 'done', 'terminal_done', 'terminal_over', 'reward_type', 'termination_reason',
                  'reward_valid', 'strict_success', 'policy_reward', 'steps', 'assistant_tokens',
                  'guard_rejections', 'action_attempts', 'infrastructure_invalid', 'reward_unverifiable')
        if any(any(k not in row or k not in tasks[row['task_id']] for k in fields) for row in infos):
            raise ValueError('GPU validation is missing behavioral evidence')
        mismatches = [row['task_id'] for row in infos if any(row.get(k) != tasks[row['task_id']].get(k) for k in fields)]
        if mismatches:
            raise ValueError(f'GPU validation differs at step{step}: {mismatches}')
        comparisons.append({'step': step, 'tasks': 50, 'adapter_sha256': digest, 'exact_taskwise_agreement': True})
    return {'passed': True, 'comparisons': comparisons, 'pilot_updates': 1,
            'scope': 'Both initial912 and actual post-update weights match independent serial eager validation; not an efficacy test'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'parent-reference', 'step1-reference', 'step1-adapter', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.pilot, args.parent_reference, args.step1_reference, args.step1_adapter)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
