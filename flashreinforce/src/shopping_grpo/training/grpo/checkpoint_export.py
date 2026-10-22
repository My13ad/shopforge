"""Extract validated LoRA tensors from a single-rank checkpoint on CPU."""
from __future__ import annotations


def extract_adapter_tensors(state: dict, expected_shapes: dict) -> dict:
    selected = {}
    for original, value in state.items():
        key = original.replace(".lora_A.default.", ".lora_A.").replace(
            ".lora_B.default.", ".lora_B.")
        if key not in expected_shapes:
            if "lora_" in key:
                raise ValueError(f"unexpected adapter tensor: {original}")
            continue
        if key in selected:
            raise ValueError(f"duplicate adapter tensor after normalization: {key}")
        # The entry point rejects world_size != 1; local DTensors are then complete.
        if hasattr(value, "to_local"):
            value = value.to_local()
        if list(value.shape) != list(expected_shapes[key]):
            raise ValueError(f"adapter shape differs from parent: {key}")
        selected[key] = value.detach().cpu().contiguous()
    if set(selected) != set(expected_shapes):
        raise ValueError("checkpoint is missing parent adapter tensors")
    return selected
