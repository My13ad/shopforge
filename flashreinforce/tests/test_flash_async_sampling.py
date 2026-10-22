import asyncio
from concurrent.futures import Future
from threading import Event
from types import SimpleNamespace

import pytest

from shopping_grpo.training.grpo.async_sampling import CompletionPool, Submitted, resume_generation


def test_completions_bypass_a_slow_tail_and_continue_during_learning():
    submitted, futures, cancelled = Event(), {}, []
    def submit(item):
        future = Future(); futures[item] = future
        if len(futures) == 4:submitted.set()
        return Submitted(future, lambda: cancelled.append(item))
    pool = CompletionPool(range(12), submit, inflight=4, capacity=4, timeout=2)
    try:
        assert submitted.wait(2)
        futures[2].set_result('second'); futures[1].set_result('first')
        assert next(pool)[:2] == (2, 'second')
        assert next(pool)[:2] == (1, 'first')
        # No learner call is needed to keep the persistent slow tail alive.
        assert not futures[0].done() and not futures[3].done()
        for _ in range(1000):
            if 4 in futures:break
            Event().wait(.001)
        assert 4 in futures
        futures[4].set_result('next-batch')
        assert next(pool)[:2] == (4, 'next-batch')
    finally:
        pool.close()
    assert 0 in cancelled and 3 in cancelled


def test_pool_drains_all_items_once_and_propagates_worker_failure():
    def submit(item):
        future = Future();future.set_result(item)
        return Submitted(future, future.cancel)
    pool = CompletionPool(range(12), submit, inflight=4, capacity=4, timeout=2)
    try:
        assert [item for item, output, seconds in pool] == list(range(12))
        assert pool.metrics()['flash_async/delivered'] == 12
    finally:pool.close()
    def fail(item):
        future = Future();future.set_exception(RuntimeError('native rollout failed'))
        return Submitted(future, future.cancel)
    pool = CompletionPool([0], fail, inflight=1, capacity=1, timeout=2)
    try:
        with pytest.raises(RuntimeError, match='native rollout failed'):next(pool)
    finally:pool.close()


def test_backpressure_bounds_finished_and_inflight_trajectories():
    called, submitted = Event(), []
    def submit(item):
        submitted.append(item);future=Future();future.set_result(item)
        if len(submitted) == 5:called.set()
        return Submitted(future, future.cancel)
    pool = CompletionPool(range(10000), submit, inflight=4, capacity=2, timeout=2)
    try:
        assert called.wait(2)
        assert len(submitted) <= 6
        assert pool.metrics()['flash_async/queued'] == 2
    finally:pool.close()


def test_partial_turn_preserves_old_and_new_tokens_probabilities_and_budget():
    calls=[]
    async def generate(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            return SimpleNamespace(token_ids=[10,11], log_probs=[-1.,-2.], num_preempted=0,
                stop_reason='aborted', extra_fields={'sampling_policy_version':0})
        return SimpleNamespace(token_ids=[12], log_probs=[-.5], num_preempted=0,
            stop_reason='completed', extra_fields={'sampling_policy_version':1})
    trace={'versions':[],'interruptions':0}
    output=asyncio.run(resume_generation(generate,[5,6],{'max_tokens':4,'temperature':1},
        make_output=SimpleNamespace,trace=trace))
    assert output.token_ids==[10,11,12] and output.log_probs==[-1.,-2.,-.5]
    assert trace=={'versions':[0,0,1],'interruptions':1}
    assert calls[1]['prompt_ids']==[5,6,10,11] and calls[1]['sampling_params']['max_tokens']==2


def test_native_empty_completion_reaches_existing_whole_batch_admission():
    async def generate(**kwargs):
        return SimpleNamespace(token_ids=[],log_probs=None,num_preempted=0,
            stop_reason='completed',extra_fields={'sampling_policy_version':0})
    trace={'versions':[],'interruptions':0}
    output=asyncio.run(resume_generation(generate,[5],{'max_tokens':4},
        make_output=SimpleNamespace,trace=trace))
    assert output.token_ids==[] and output.log_probs==[]
    assert trace=={'versions':[],'interruptions':0}
    assert output.extra_fields=={'sampling_policy_versions':[]}


@pytest.mark.parametrize('logps,version',[([],0),([-1.],None)])
def test_partial_turn_rejects_missing_sampler_evidence(logps,version):
    async def generate(**kwargs):
        return SimpleNamespace(token_ids=[10],log_probs=logps,num_preempted=0,
            stop_reason='completed',extra_fields={'sampling_policy_version':version})
    with pytest.raises(ValueError):
        asyncio.run(resume_generation(generate,[5],{'max_tokens':4},make_output=SimpleNamespace))


def test_actual_server_adapter_drains_old_requests_and_resumes_under_published_version():
    import ast
    from pathlib import Path
    from shopping_grpo.training.grpo.async_sampling import GenerationGate
    path=Path(__file__).resolve().parents[1]/'src/shopping_grpo/training/grpo/async_runtime.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AsyncVLLMServer')
    class NativeServer:
        def _post_init(self, devices):pass
        async def generate(self, **kwargs):
            if self.global_steps==0:
                self.old_started.set();await self.abort_event.wait()
                return SimpleNamespace(token_ids=[10,11],log_probs=[-1.,-2.],num_preempted=0,
                    stop_reason='aborted',extra_fields={})
            return SimpleNamespace(token_ids=[12],log_probs=[-.5],num_preempted=0,
                stop_reason='completed',extra_fields={})
        async def abort_all_requests(self):
            self.abort_event.set();return {'aborted_count':1}
        async def resume_generation(self):pass
    namespace={'vLLMHttpServer':NativeServer,'GenerationGate':GenerationGate}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    async def exercise():
        server=namespace['AsyncVLLMServer']();server.global_steps=0
        server.abort_event=asyncio.Event();server.old_started=asyncio.Event();server._post_init('0')
        trace={'versions':[],'interruptions':0}
        task=asyncio.create_task(resume_generation(server.generate,[5],{'max_tokens':4},
            make_output=SimpleNamespace,trace=trace))
        await server.old_started.wait()
        await server.flash_pause()
        # The old part has returned, but its continuation is blocked before
        # taking a version until weights have been published and resumed.
        await asyncio.sleep(0)
        assert not task.done() and server.flash_gate.active==0
        server.global_steps=1
        await server.flash_resume(1)
        output=await asyncio.wait_for(task,2)
        assert output.token_ids==[10,11,12] and output.log_probs==[-1.,-2.,-.5]
        assert trace=={'versions':[0,0,1],'interruptions':1}
        assert server.flash_gate.active==0
    asyncio.run(exercise())
