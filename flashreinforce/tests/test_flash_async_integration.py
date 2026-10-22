from copy import deepcopy

import pytest

from shopping_grpo.training.grpo import flashreinforce as flash
from test_flashreinforce import async_config, batch, config, args, entry, parent  # noqa: F401
from test_flash_batch_admission import candidate, native_harness


def test_actual_native_fit_consumes_completed_batches_and_runs_real_cpu_optimizer(monkeypatch,tmp_path):
    # Each prompt is independent, even when completions mix dispatched cohorts.
    candidates=[]
    for index in range(6):
        data=candidate(invalid=(index==0))
        for info in data.non_tensor_batch['shopping']:info['task_id']+=4*index
        candidates.append(data)
    run=native_harness(monkeypatch,tmp_path,candidates,target=3,execution='async')
    run.trainer.fit()
    assert len(run.observed)==3
    assert run.optimizer.state[run.parameter]['step']==3 and run.scheduler.last_epoch==3
    assert run.weights==[0,1,2,3]
    assert run.trainer._shopping_flash_pool.closed
    assert all(len(data)==4 for data in run.observed)
    assert all(data.meta_info['flash_learner_version']==i for i,data in enumerate(run.observed))


@pytest.mark.parametrize('mutation',['future','missing','wrong_count'])
def test_async_batch_rejects_invented_or_misaligned_behavior_versions(mutation):
    settings=async_config(config());data=batch();data.meta_info['flash_learner_version']=2
    for row,info in enumerate(data.non_tensor_batch['shopping']):
        info['sampling_policy_versions']=[0]*int(data.batch['response_mask'][row].sum())
    flash.validate_flashreinforce_batch(data,settings)
    if mutation=='future':data.non_tensor_batch['shopping'][0]['sampling_policy_versions']=[3]
    elif mutation=='missing':del data.non_tensor_batch['shopping'][0]['sampling_policy_versions']
    else:data.non_tensor_batch['shopping'][0]['sampling_policy_versions']=[0,0]
    with pytest.raises(ValueError,match='provenance'):flash.validate_flashreinforce_batch(data,settings)


def test_async_entry_selects_resident_sampling_and_953_smoke_pool(monkeypatch,parent,tmp_path):
    options=args(monkeypatch,parent,tmp_path,method='flashreinforce',profile='smoke',flash_execution='async')
    command,environment=entry.build_command(options)
    assert environment['GRPO_TRAIN_FILE'].endswith('train953.parquet')
    assert 'shopping_flashreinforce.execution=async' in command
    assert 'shopping_flashreinforce.async_inflight=32' in command
    assert 'actor_rollout_ref.rollout.free_cache_engine=false' in command
    assert 'trainer.total_training_steps=4' in command and 'trainer.save_freq=4' in command
    assert any('AsyncCheckpointManager' in value for value in command)
    assert any('agent_loop_async.yaml' in value for value in command)


def test_async_recipe_composes_against_real_pinned_verl_hydra(monkeypatch,parent,tmp_path):
    from hydra import compose,initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    options=args(monkeypatch,parent,tmp_path,method='flashreinforce',profile='smoke',flash_execution='async')
    command,environment=entry.build_command(options)
    for key,value in environment.items():monkeypatch.setenv(key,value)
    GlobalHydra.instance().clear()
    search=(root/'outputs/preparation/v1/runtime-wheel/source/verl/trainer/config').as_posix()
    with initialize_config_dir(version_base=None,config_dir=str(root/'configs')):
        recipe=compose(config_name='grpo',overrides=command[5:]+[f'hydra.searchpath=["file://{search}"]'])
    flash.validate_flashreinforce_config(recipe)
    assert recipe.shopping_flashreinforce.async_queue_batches==8
    assert recipe.shopping_flashreinforce.async_inflight==32
    assert recipe.trainer.total_training_steps==4 and not recipe.actor_rollout_ref.rollout.free_cache_engine


def test_async_shop_adapter_matches_native_response_prefix_truncation():
    import ast,asyncio
    from pathlib import Path
    from types import SimpleNamespace
    from shopping_grpo.training.grpo.async_sampling import policy_trace
    path=Path(__file__).resolve().parents[1]/'src/shopping_grpo/training/grpo/adapter/async_agent_loop.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    class Base:
        context_compaction_enable=False
        response_length=4
        async def run(self,*args,**kwargs):
            policy_trace.get()['versions'].extend([0,0,1,1])
            return SimpleNamespace(response_mask=[1,0,1,1],extra_fields={'shopping':{}})
    namespace={'ShoppingToolAgentLoop':Base,'policy_trace':policy_trace}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    output=asyncio.run(namespace['AsyncShoppingToolAgentLoop']().run({}))
    assert output.extra_fields['shopping']['sampling_policy_versions']==[0,0,1]
    assert policy_trace.get() is None


def test_async_native_tensordict_conversion_keeps_sample_versions_out_of_batch_metadata(monkeypatch,tmp_path):
    import ast,__future__
    from pathlib import Path
    import numpy as np
    import torch
    root=Path(__file__).resolve().parents[1]
    source=root/'outputs/preparation/v1/runtime-wheel/source/verl/utils/tensordict_utils.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_tensordict')
    class Stack(list):
        @classmethod
        def from_list(cls,items):return cls(items)
    namespace={'torch':torch,'np':np,'NonTensorStack':Stack,'NonTensorData':lambda v:v,
               'TensorDict':lambda **kwargs:kwargs}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec',
                 flags=__future__.annotations.compiler_flag),namespace)
    run=native_harness(monkeypatch,tmp_path,[candidate(),candidate()],target=1,execution='async')
    worker=run.trainer.async_rollout_manager.agent_loop_workers[0]
    remote=worker.generate_sequences.remote
    def actual_sampler_fields(data):
        ref=remote(data)
        ref.result().non_tensor_batch['global_steps']=np.zeros(len(data),dtype=object)
        return ref
    monkeypatch.setattr(worker.generate_sequences,'remote',actual_sampler_fields)
    original=run.trainer._update_actor
    def convert_and_update(data):
        # Execute the actual pinned conversion function on the native loop's
        # final batch, including the sampler's real per-row global_steps field.
        namespace['get_tensordict']({**data.batch,**data.non_tensor_batch},data.meta_info)
        return original(data)
    monkeypatch.setattr(run.trainer,'_update_actor',convert_and_update)
    run.trainer.fit()
    assert run.optimizer.state[run.parameter]['step']==1
