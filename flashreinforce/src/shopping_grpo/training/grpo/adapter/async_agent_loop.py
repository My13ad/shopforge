"""Attach per-policy-token provenance to the unchanged Shop trajectory lifecycle."""
from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
from shopping_grpo.training.grpo.async_sampling import policy_trace


class AsyncShoppingToolAgentLoop(ShoppingToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        if self.context_compaction_enable:
            raise ValueError("async token provenance requires uncompacted response offsets")
        trace = {"versions": [], "interruptions": 0}
        token = policy_trace.set(trace)
        try:
            output = await super().run(sampling_params, **kwargs)
            count = sum(output.response_mask)
            if len(trace["versions"]) > count and len(output.response_mask) == self.response_length:
                # The native ToolAgentLoop truncates the response prefix at its
                # hard cap. Apply exactly that existing prefix cut to metadata.
                trace["versions"] = trace["versions"][:count]
            if len(trace["versions"]) != count:
                raise ValueError("async sampling provenance no longer matches the policy mask")
            output.extra_fields.pop("sampling_policy_versions", None)
            shopping = output.extra_fields["shopping"]
            shopping["sampling_policy_versions"] = trace["versions"]
            shopping["sampling_interruptions"] = trace["interruptions"]
            if trace["versions"]:
                output.extra_fields["min_global_steps"] = min(trace["versions"])
                output.extra_fields["max_global_steps"] = max(trace["versions"])
            return output
        finally:
            policy_trace.reset(token)
