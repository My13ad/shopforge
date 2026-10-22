"""An already exported adapter must remain a valid parent for another export."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file


@pytest.mark.parametrize("adapter_only", [False, True])
def test_reexport_does_not_hash_inherited_export_manifest(tmp_path, monkeypatch, adapter_only):
    script = Path(__file__).resolve().parents[1] / "scripts/export_grpo_adapter.py"
    spec = importlib.util.spec_from_file_location("checkpoint_export_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path))
    parent = tmp_path / "sft-parent"
    parent.mkdir()
    save_file({"q.lora_A.weight": torch.ones(2, 4)}, parent / "adapter_model.safetensors")
    (parent / "adapter_config.json").write_text('{"r":16,"lora_alpha":32}')
    (parent / "tokenizer_config.json").write_text('{"chat_template":"fixture"}')
    (parent / "chat_template.jinja").write_text("fixture template")
    checkpoint = tmp_path / "actor"
    checkpoint.mkdir()
    (checkpoint / "fsdp_config.json").write_text('{"world_size":1}')
    (checkpoint / "optim_world_size_1_rank_0.pt").write_bytes(b"optimizer receipt fixture")
    torch.save({"q.lora_A.default.weight": torch.full((2, 4), 2.0),
                "q.base_layer.weight": torch.zeros(4, 4)},
               checkpoint / "model_world_size_1_rank_0.pt")
    # The fixture has ordinary CPU tensors. Distributed rendezvous is outside
    # this regression's copy/manifest path and requires no process group.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    for name in ("first-export", "second-export"):
        destination = tmp_path / name
        monkeypatch.setattr(sys, "argv", [str(script), "--checkpoint", str(checkpoint),
            "--parent-adapter", str(parent), "--output", str(destination)] + (["--adapter-only"] if adapter_only else []))
        module.main()
        manifest = json.loads((destination / "export-manifest.json").read_text())
        assert "export-manifest.json" not in manifest["files"]
        assert all(hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
                   for name, digest in manifest["files"].items())
        assert (destination / "chat_template.jinja").read_text() == "fixture template"
        assert manifest["native_checkpoint_preserved"]
        assert (destination / "optim_world_size_1_rank_0.pt").exists() is not adapter_only
        parent = destination
