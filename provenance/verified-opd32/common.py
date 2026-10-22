"""Small single-GPU OPD pilot: validation, serialization, and sampled loss."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def importance_loss(current_logp, old_logp, teacher_logp):
    """Immediate-token sampled reverse-KL policy-gradient surrogate, gamma=0.

    At the unchanged behavior policy, its expected gradient equals the
    conditional reverse-KL gradient. The scalar surrogate is not a KL value.
    A single optimizer update is allowed per freshly sampled policy batch.
    """
    import torch
    if current_logp.shape != old_logp.shape or old_logp.shape != teacher_logp.shape:
        raise ValueError("Student and teacher token positions must align")
    if not all(torch.isfinite(x).all() for x in [current_logp, old_logp, teacher_logp]):
        raise ValueError("Token log probabilities must be finite")
    advantage = (teacher_logp - old_logp).detach()
    ratio = (current_logp - old_logp.detach()).exp()
    return -(ratio * advantage), (old_logp - teacher_logp).detach()


def extract_teacher_scores(response, prompt_ids, sampled_ids):
    """Require the endpoint to score exactly the sampled IDs, including EOS."""
    import math
    logprobs = response["choices"][0].get("logprobs") or {}
    all_ids = prompt_ids + sampled_ids
    if logprobs.get("tokens") != [f"token_id:{x}" for x in all_ids]:
        raise ValueError("Teacher endpoint token IDs do not match the submitted sequence")
    values = logprobs.get("token_logprobs") or []
    if len(values) != len(all_ids):
        raise ValueError("Teacher endpoint returned an unexpected token count")
    scored = values[len(prompt_ids):]
    if not sampled_ids or len(scored) != len(sampled_ids) or not all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in scored
    ):
        raise ValueError("Missing or non-finite teacher response probabilities")
    return scored


def choose_training_ids(train_ids, development_ids, final_ids, excluded_ids, seed, count, prefix_ids=()):
    import random
    train = set(train_ids)
    if train & (set(development_ids) | set(final_ids)):
        raise ValueError("Training task IDs overlap evaluation")
    if count <= 0:
        raise ValueError("Training task count must be positive")
    prefix = list(prefix_ids)
    candidates = sorted(train - set(excluded_ids))
    if len(prefix) != len(set(prefix)) or not set(prefix) <= set(candidates) or len(prefix) > count:
        raise ValueError("Task prefix must contain unique eligible training IDs within the budget")
    if count > len(candidates):
        raise ValueError("Not enough eligible training tasks")
    remaining = sorted(set(candidates) - set(prefix))
    return prefix + random.Random(seed).sample(remaining, count - len(prefix))


def checkpoint_audit(adapter, previous_digest):
    import torch
    from safetensors.torch import load_file
    file = Path(adapter) / "adapter_model.safetensors"
    tensors = load_file(str(file))
    if not tensors or not all(torch.isfinite(x).all() for x in tensors.values()):
        raise ValueError("Adapter contains missing or non-finite tensors")
    digest = sha(file)
    if digest == previous_digest:
        raise ValueError("Training did not change the adapter")
    return {"passed": True, "tensor_count": len(tensors), "sha256": digest,
            "previous_sha256": previous_digest, "changed": True}
