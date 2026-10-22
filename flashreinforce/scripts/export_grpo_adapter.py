"""Export LoRA and optimizer receipts on CPU; never delete the source checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Native global_step_N/actor directory")
    parser.add_argument("--parent-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-only", action="store_true",
                        help="Retain inference weights and tokenizer without optimizer copies")
    args = parser.parse_args()
    source = args.checkpoint.resolve()
    parent, output = args.parent_adapter.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit("export output must be a new directory")
    config = json.loads((source / "fsdp_config.json").read_text())
    if config.get("world_size") != 1:
        raise SystemExit("this export supports exactly one GPU/rank")
    weights = source / "model_world_size_1_rank_0.pt"
    if not weights.is_file():
        raise SystemExit("missing native FSDP model shard")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from shopping_grpo.training.grpo.checkpoint_export import extract_adapter_tensors
    with safe_open(parent / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        shapes = {key: handle.get_slice(key).get_shape() for key in handle.keys()}
    if not shapes or not all("lora_" in key for key in shapes):
        raise SystemExit("parent contains unsupported non-LoRA trained tensors")
    initialized_here = False
    try:
        with tempfile.TemporaryDirectory() as temporary:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    "gloo", init_method="file://" + str(Path(temporary) / "group"), rank=0, world_size=1)
                initialized_here = True
            elif torch.distributed.get_world_size() != 1:
                raise SystemExit("export process belongs to a multi-rank group")
            state = torch.load(weights, map_location="cpu", weights_only=False)
            adapter = extract_adapter_tensors(state, shapes)
            output.mkdir(parents=True)
            for path in parent.iterdir():
                # A previous export may be the parent. Its manifest must not
                # become a self-hash entry when this export writes its own.
                if (path.is_file() and path.name != "export-manifest.json"
                        and path.suffix in {".json", ".jinja", ".model", ".txt"}):
                    shutil.copy2(path, output / path.name)
            save_file(adapter, output / "adapter_model.safetensors")
            for name in ("optim_world_size_1_rank_0.pt", "extra_state_world_size_1_rank_0.pt",
                         "fsdp_config.json", "lora_train_meta.json"):
                if not args.adapter_only and (source / name).is_file():
                    shutil.copy2(source / name, output / name)
            receipt = {"source_checkpoint": str(source), "native_checkpoint_preserved": True,
                       "adapter_only": args.adapter_only,
                       "native_resume_requires_source_checkpoint": True,
                       "adapter_tensors": len(adapter), "files": {}}
            for path in output.iterdir():
                with path.open("rb") as handle:
                    receipt["files"][path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
            (output / "export-manifest.json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps({"output": str(output), "adapter_tensors": len(adapter)}))
    finally:
        if initialized_here:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
