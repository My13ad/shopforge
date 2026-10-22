"""Evaluate a candidate and pair it with saved uniform-SFT development rollouts."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def diagnostic_counts(rows):
    turns = [message for row in rows for message in row.get('messages', [])
             if message.get('role') == 'assistant' and message.get('generation_metadata')]
    return {
        'tasks_with_guard_rejections': sum(bool(row.get('blocked_tool_calls')) for row in rows),
        'incomplete_function_turns': sum(message['generation_metadata']['incomplete_function'] for message in turns),
        'generation_limit_turns': sum(message['generation_metadata'].get('max_tokens_reached', False) for message in turns),
        'generated_turns': len(turns),
        'generated_tokens': sum(message['generation_metadata'].get('completion_tokens', 0) for message in turns),
        'error_type_counts': dict(Counter((row.get('error') or {}).get('type') for row in rows if row.get('error'))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-evaluation', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--base-url', default='http://127.0.0.1:5700')
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'phase': 'preparing', 'started_at': now(), 'pid': os.getpid(),
              'expected_tasks_per_actor': args.count, 'completed': {'reference': 0, 'sft': 0}}
    write_json(output/'status.json', status)
    try:
        train_path = workspace/'data/sft/train.jsonl'
        validation_path = workspace/'data/sft/validation.jsonl'
        train_ids = {row['task_id'] for row in read_rows(train_path)}
        validation_rows = read_rows(validation_path)
        ids = [row['task_id'] for row in validation_rows]
        assert len(ids) == len(set(ids)) == 200 and not (set(ids) & train_ids)
        if not 1 <= args.count <= 200:
            raise ValueError('count must be within 1..200')
        tasks = [{'task_id': task_id} for task_id in ids[:args.count]]
        reference_config = json.loads((args.reference_evaluation/'run_config.json').read_text())
        reference_rows = read_rows(args.reference_evaluation/'sft-trajectories.jsonl')
        assert [row['task_id'] for row in reference_rows] == ids
        reference_rows = reference_rows[:args.count]
        assert all(row.get('actor') == 'sft' for row in reference_rows)
        benchmark = output/'benchmark.jsonl'
        benchmark.write_text(''.join(json.dumps(task)+'\n' for task in tasks), encoding='utf-8')
        shutil.copytree(workspace/'src', output/'source/src', ignore=shutil.ignore_patterns('__pycache__'))
        driver = output/'source/round_task_eval.py'
        shutil.copy2(__file__, driver)
        sys.path.insert(0, str(output/'source/src'))
        from shopping_grpo.evaluation.blind_guard import guard_blind_final
        from shopping_grpo.evaluation.local_inference import TransformersTransport
        from shopping_grpo.evaluation.rollout import (
            OpenAIChatClient, collect_for_task, append_jsonl, _is_infrastructure_failure,
        )
        from shopping_grpo.evaluation.summary import summarize_trajectories
        guard_blind_final([benchmark], allowed=False)
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        from peft import PeftModel
        torch.manual_seed(42)
        processor = AutoProcessor.from_pretrained(args.adapter, local_files_only=True)
        assert processor.chat_template
        base = AutoModelForMultimodalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True,
        )
        model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).to('cuda').eval()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        linear_layer = next(module for module in model.modules() if isinstance(module, Qwen3_5GatedDeltaNet))
        kernel_names = {
            name: (getattr(linear_layer, name).__module__ + '.' + getattr(linear_layer, name).__name__)
            if getattr(linear_layer, name) is not None else None
            for name in ['causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule', 'recurrent_gated_delta_rule']
        }
        package_names = ['torch', 'triton', 'transformers', 'peft', 'flash-linear-attention', 'fla-core', 'einops']
        package_versions = {}
        for name in package_names:
            try:
                package_versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                package_versions[name] = None
        transport = TransformersTransport(model, processor)
        protocol = {
            'split': 'SFT validation development set; not the frozen final test',
            'backend': 'Transformers + PEFT, candidate with saved uniform-SFT reference',
            'max_steps': 35, 'max_tokens': 512, 'temperature': 0.0, 'top_p': 1.0,
            'context_window': 24576, 'context_safety_margin': 512, 'context_compaction': False,
            'observation_token_budget': 1536, 'observation_detail_token_budget': 4096,
            'observation_generic_token_budget': 768, 'observation_search_top_k': 20,
            'environment': 'shopsimulator-environment-v2.1', 'reward': 'shopsimulator-reward-v3',
            'rollouts_per_task_per_actor': 1, 'seed': 42,
            'tool_parser': 'Qwen XML completed-function parser; truncated functions not repaired',
            'stop_token_ids': transport.eos_token_ids,
            'stop_policy': 'Document EOS and Qwen <|im_end|> assistant boundary',
        }
        config = {
            'model': str(args.model.resolve()), 'adapter': str(args.adapter.resolve()),
            'validation_sha256': hashlib.sha256(validation_path.read_bytes()).hexdigest(),
            'train_sha256': hashlib.sha256(train_path.read_bytes()).hexdigest(),
            'benchmark_sha256': hashlib.sha256(benchmark.read_bytes()).hexdigest(),
            'adapter_sha256': hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
            'chat_template_sha256': hashlib.sha256(processor.chat_template.encode()).hexdigest(),
            'driver_sha256': hashlib.sha256(driver.read_bytes()).hexdigest(),
            'packages': package_versions,
            'linear_attention_kernels': kernel_names,
            'gpu': torch.cuda.get_device_name(),
            'protocol': protocol,
            'reference_evaluation': str(args.reference_evaluation.resolve()),
            'reference_adapter_sha256': reference_config['adapter_sha256'],
        }
        for key in ['validation_sha256', 'train_sha256', 'benchmark_sha256', 'chat_template_sha256',
                    'packages', 'linear_attention_kernels', 'gpu']:
            assert config[key] == reference_config[key], f'Reference mismatch: {key}'
        for key, value in protocol.items():
            if key != 'backend':
                assert value == reference_config['protocol'][key], f'Protocol mismatch: {key}'
        for relative in ['evaluation/rollout.py', 'evaluation/summary.py', 'evaluation/local_inference.py',
                         'environment/client.py', 'environment/observation.py', 'environment/tools.py']:
            saved = args.reference_evaluation/'source/src/shopping_grpo'/relative
            current = output/'source/src/shopping_grpo'/relative
            assert saved.read_bytes() == current.read_bytes(), f'Collector mismatch: {relative}'
        write_json(output/'run_config.json', config)
        if args.preflight_only:
            status.update(phase='completed', configuration_only=True, finished_at=now())
            write_json(output/'status.json', status)
            print('EVALUATION_CONFIGURATION_PASSED', flush=True)
            return
        records = {'reference': reference_rows, 'sft': []}
        expected_ids = [task['task_id'] for task in tasks]
        reference_summary = summarize_trajectories(expected_ids, reference_rows)
        write_json(output/'reference-summary.json', reference_summary)
        status['completed']['reference'] = len(reference_rows)
        clients = {
            actor: OpenAIChatClient(
                model=f'qwen35-{actor}', base_url='local-transformers', api_key='LOCAL',
                temperature=0.0, top_p=1.0, max_tokens=512, context_window=24576,
                context_safety_margin=512, context_compaction_enable=False,
                observation_token_budget=1536, observation_detail_token_budget=4096,
                observation_generic_token_budget=768, observation_search_top_k=20,
                token_counter=transport.count_chat_tokens,
                observation_token_counter=transport.count_text_tokens,
                transport=transport,
            ) for actor in ['sft']
        }
        status['phase'] = 'evaluating'
        for index, task in enumerate(tasks, 1):
            for actor in ['sft']:
                status.update(active_actor=actor, active_task_id=task['task_id'], updated_at=now())
                write_json(output/'status.json', status)
                transport.actor = actor
                task_started = time.monotonic()
                trajectory = collect_for_task(
                    task, client=clients[actor], base_url=args.base_url, max_steps=35,
                )
                trajectory['actor'] = actor
                trajectory['elapsed_seconds'] = round(time.monotonic()-task_started, 3)
                append_jsonl(output/f'{actor}-trajectories.jsonl', [trajectory])
                records[actor].append(trajectory)
                summary = summarize_trajectories(expected_ids, records[actor])
                summary.update(protocol=protocol, diagnostics=diagnostic_counts(records[actor]))
                write_json(output/f'{actor}-summary.json', summary)
                status['completed'][actor] = index
                status['elapsed_minutes'] = round((time.monotonic()-started)/60, 2)
                status['strict_successes'] = {
                    key: summarize_trajectories(expected_ids, rows)['strict_successes']
                    for key, rows in records.items()
                }
                status['peak_torch_memory_gib'] = round(torch.cuda.max_memory_allocated()/1024**3, 3)
                write_json(output/'status.json', status)
                detail = (trajectory.get('terminal_result') or {}).get('reward_detail') or {}
                print('TASK',actor,index,args.count,'id',task['task_id'],'status',trajectory['status'],
                      'outcome',detail.get('reward_type'),'steps',len(trajectory['steps']),
                      'seconds',trajectory['elapsed_seconds'],flush=True)
                if _is_infrastructure_failure(trajectory):
                    raise RuntimeError('Infrastructure failure: '+json.dumps(trajectory.get('error')))
                if trajectory.get('error') and trajectory['error']['type'] != 'ContextBudgetError':
                    raise RuntimeError('Unexpected rollout error: '+json.dumps(trajectory['error']))
        successes = {
            actor: set(summarize_trajectories(expected_ids, rows)['strict_success_task_ids'])
            for actor, rows in records.items()
        }
        paired = {
            'expected_tasks': len(tasks), 'both_success': sorted(successes['reference'] & successes['sft']),
            'candidate_only_success': sorted(successes['sft']-successes['reference']),
            'reference_only_success': sorted(successes['reference']-successes['sft']),
            'both_failure': sorted(set(expected_ids)-(successes['reference'] | successes['sft'])),
            'strict_success_rate_difference': (len(successes['sft'])-len(successes['reference']))/len(tasks),
            'reference': 'Original-data uniform SFT, final adapter at step 300',
            'candidate_adapter': str(args.adapter.resolve()),
            'protocol': protocol,
        }
        write_json(output/'paired-comparison.json', paired)
        badcases = []
        for row in records['sft']:
            if row['task_id'] in successes['sft']:
                continue
            detail = (row.get('terminal_result') or {}).get('reward_detail') or {}
            badcases.append({
                'task_id': row['task_id'], 'status': row['status'],
                'reward_type': detail.get('reward_type'), 'final_reward': row['final_reward'],
                'query': next((message['content'] for message in row['messages'] if message['role']=='user'), ''),
                'blocked_reasons': [item['reason'] for item in row.get('blocked_tool_calls', [])],
                'actions': [{'tool': step['tool_name'], 'parameters': step['parameters']} for step in row['steps']],
                'error': row.get('error'),
            })
        write_json(output/'sft-badcases.json', badcases)
        status.update(phase='completed', active_actor=None, active_task_id=None,
                      finished_at=now(), elapsed_minutes=round((time.monotonic()-started)/60, 2))
        write_json(output/'status.json', status)
        print('EVALUATION_COMPLETED',json.dumps(status),flush=True)
    except Exception as error:
        status.update(phase='failed', error=f'{type(error).__name__}: {error}', finished_at=now())
        write_json(output/'status.json', status)
        raise


if __name__ == '__main__':
    main()
