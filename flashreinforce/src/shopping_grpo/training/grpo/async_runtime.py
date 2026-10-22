"""veRL adapters for resident vLLM, version-safe refits, and partial assistant turns."""
from functools import wraps

import ray

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from shopping_grpo.training.grpo.async_sampling import GenerationGate, policy_trace, resume_generation


class AsyncVLLMServer(vLLMHttpServer):
    def _post_init(self, cuda_visible_devices):
        super()._post_init(cuda_visible_devices)
        self.flash_gate = GenerationGate()

    async def generate(self, *args, **kwargs):
        # The native server reads global_steps after generation. Capture before
        # generation and drain these readers before allowing a weight refit.
        version = await self.flash_gate.enter(lambda: self.global_steps)
        try:
            output = await super().generate(*args, **kwargs)
            output.extra_fields["sampling_policy_version"] = version
            return output
        finally:
            await self.flash_gate.leave()

    async def flash_pause(self):
        async def abort():
            report = await self.abort_all_requests()
            if report.get("error"):
                raise RuntimeError(report["error"])
        await self.flash_gate.pause(abort)

    async def flash_resume(self, version):
        if self.global_steps != version:
            raise RuntimeError("refit did not publish the expected sampling version")
        await self.flash_gate.resume(self.resume_generation)


class AsyncVLLMReplica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(AsyncVLLMServer)


class VersionedLLMClient(LLMServerClient):
    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        async def generate(**part):
            return await super(VersionedLLMClient, self).generate(request_id, **part, **kwargs)
        return await resume_generation(generate, prompt_ids, sampling_params,
                                       make_output=TokenOutput, trace=policy_trace.get())


class AsyncAgentLoopManager(AgentLoopManager):
    def __init__(self, config, llm_client, *args, **kwargs):
        client = VersionedLLMClient(config, load_balancer_handle=llm_client._load_balancer)
        super().__init__(config, client, *args, **kwargs)


class AsyncCheckpointManager(CheckpointEngineManager):
    """Keep inference resident during learning; refits alone interrupt generation."""

    sampling_version = 0

    @auto_await
    async def sleep_replicas(self):
        return

    @auto_await
    async def update_weights(self, global_steps=None):
        import asyncio
        servers = [server for replica in self.replicas for server in replica.servers]
        await asyncio.gather(*[server.flash_pause.remote() for server in servers])
        # A failed refit stays paused and fails the run; never resume mixed weights.
        await super().update_weights(global_steps)
        self.sampling_version = global_steps
        await asyncio.gather(*[server.flash_resume.remote(global_steps) for server in servers])


def activate_runtime():
    original = LLMServerManager.__init__
    if getattr(original, "_shopping_flash_async", False):
        return

    @wraps(original)
    def init(self, config, *args, **kwargs):
        original(self, config, *args, **kwargs)
        if config.get("shopping_flashreinforce", {}).get("execution", "sync") == "async":
            self.rollout_replica_class = AsyncVLLMReplica

    init._shopping_flash_async = True
    LLMServerManager.__init__ = init
