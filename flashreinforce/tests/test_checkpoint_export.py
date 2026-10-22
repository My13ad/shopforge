import pytest
import torch
from shopping_grpo.training.grpo.checkpoint_export import extract_adapter_tensors


def test_extracts_only_lora_and_removes_default_adapter_name():
    state = {"base.q.lora_A.default.weight": torch.ones(2, 4),
             "base.q.lora_B.default.weight": torch.ones(4, 2),
             "base.q.base_layer.weight": torch.zeros(4, 4)}
    expected = {"base.q.lora_A.weight": [2, 4], "base.q.lora_B.weight": [4, 2]}
    result = extract_adapter_tensors(state, expected)
    assert set(result) == set(expected)
    assert all(t.device.type == "cpu" for t in result.values())
    assert len(state) == 3


@pytest.mark.parametrize("failure", ["missing", "shape", "extra", "duplicate"])
def test_bad_checkpoint_is_rejected(failure):
    expected = {"q.lora_A.weight": [2, 4]}
    state = {"q.lora_A.default.weight": torch.ones(2, 4)}
    if failure == "missing":
        state.clear()
    elif failure == "shape":
        state["q.lora_A.default.weight"] = torch.ones(4, 4)
    elif failure == "extra":
        state["other.lora_A.weight"] = torch.ones(2, 4)
    else:
        state["q.lora_A.weight"] = torch.ones(2, 4)
    with pytest.raises(ValueError):
        extract_adapter_tensors(state, expected)
