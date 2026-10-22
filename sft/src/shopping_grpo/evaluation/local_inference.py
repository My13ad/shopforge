"""Local Transformers transport for the existing deterministic rollout client.

Both Base and SFT use the same tokenizer, generation settings and XML parser.
Only complete Qwen function blocks become tool calls; truncated output remains
visible in the trajectory and is never completed on the model's behalf.
"""

from contextlib import nullcontext
import json
import re
import time
from uuid import uuid4

from shopping_grpo.training.sft.dataset import normalize_messages_for_chat_template


_FUNCTION = re.compile(r"<function=([^>]*)>(.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))", re.DOTALL,
)


def parse_qwen_completion(text):
    """Interpret Qwen XML without changing names, values or choosing an action."""
    raw = str(text)
    matches = list(_FUNCTION.finditer(raw))
    calls = []
    for match in matches:
        parameters = {}
        for parameter in _PARAMETER.finditer(match.group(2)):
            value = parameter.group(2)
            if value.startswith("\n"):
                value = value[1:]
            if value.endswith("\n"):
                value = value[:-1]
            parameters[parameter.group(1)] = value
        calls.append({
            "type": "function", "id": "call_" + uuid4().hex,
            "function": {"name": match.group(1),
                         "arguments": json.dumps(parameters, ensure_ascii=False)},
        })
    prefix = raw[:matches[0].start()] if matches else raw
    if matches:
        prefix = prefix.split("<tool_call>", 1)[0]
    content = prefix.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip("\n")
    malformed = ("<function=" in raw and len(matches) != raw.count("<function="))
    return {"role": "assistant", "content": content, "tool_calls": calls,
            "generation_metadata": {"raw_completion": raw, "incomplete_function": malformed}}


def chat_eos_token_ids(tokenizer):
    """Stop Qwen Base generation at the assistant boundary used by its chat template."""
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_id is None or end_id == tokenizer.unk_token_id:
        raise ValueError("Qwen chat tokenizer is missing <|im_end|>")
    return sorted({token_id for token_id in [tokenizer.eos_token_id, end_id] if token_id is not None})


class TransformersTransport:
    """Supply chat completions and token counts without an external model API."""

    def __init__(self, model, processor):
        import torch
        self.torch = torch
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.eos_token_ids = chat_eos_token_ids(self.tokenizer)
        self.actor = "sft"

    def render(self, messages, tools):
        normalized = normalize_messages_for_chat_template(messages)
        if normalized is None:
            raise ValueError("Conversation contains invalid JSON function arguments")
        return self.processor.apply_chat_template(
            normalized, tools=tools, tokenize=False, add_generation_prompt=True,
        )

    def count_chat_tokens(self, messages, tools):
        return len(self.tokenizer(self.render(messages, tools), add_special_tokens=False)["input_ids"])

    def count_text_tokens(self, text):
        return len(self.tokenizer(str(text), add_special_tokens=False)["input_ids"])

    def __call__(self, url, payload, headers, timeout):
        del url, headers, timeout
        if payload.get("model") != f"qwen35-{self.actor}":
            raise ValueError("Requested actor does not match the transport's active adapter mode")
        if payload.get("temperature") != 0.0 or payload.get("top_p") != 1.0:
            raise ValueError("This evaluation transport requires deterministic generation")
        prompt = self.render(payload["messages"], payload["tools"])
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        context = self.model.disable_adapter() if self.actor == "base" else nullcontext()
        started = time.monotonic()
        with context, self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=int(payload["max_tokens"]), do_sample=False,
                use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.eos_token_ids,
            )
        prompt_tokens = inputs["input_ids"].shape[1]
        generated = output[0, prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        message = parse_qwen_completion(text)
        message["generation_metadata"].update({
            "actor": self.actor, "prompt_tokens": prompt_tokens,
            "completion_tokens": int(generated.shape[0]),
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "max_tokens_reached": generated.shape[0] >= int(payload["max_tokens"]),
        })
        return {"choices": [{"message": message}], "usage": {
            "prompt_tokens": prompt_tokens, "completion_tokens": int(generated.shape[0]),
        }}
