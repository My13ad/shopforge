"""Read back unfused vLLM parameters after the original native weight loader."""
import os

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from shopping_grpo.training.grpo.diagnostics import samples_equal, weight_sample, write_receipt

_original_bucket = vLLMColocateWorkerExtension._update_weights
_original_sync = vLLMColocateWorkerExtension.update_weights_from_ipc


def _audit_bucket(self, weights, *args, **kwargs):
    weights = list(weights)
    samples = [sample for name, value in weights if (sample := weight_sample(name, value)) is not None]
    result = _original_bucket(self, weights, *args, **kwargs)
    self._shopping_received_samples.extend(samples)
    return result


def _audit_sync(self, *args, **kwargs):
    self._shopping_received_samples = []
    result = _original_sync(self, *args, **kwargs)
    loaded = [sample for name, value in self.model_runner.model.named_parameters()
              if (sample := weight_sample(name, value)) is not None]
    incoming = {item["name_suffix"]: item for item in self._shopping_received_samples}
    readback = {item["name_suffix"]: item for item in loaded}
    if not incoming or set(incoming) != set(readback) or any(
            not samples_equal(item, readback[name]) for name, item in incoming.items()):
        raise RuntimeError("vLLM merged-weight readback does not match received samples")
    write_receipt(os.environ["SHOPPING_DIAGNOSTIC_DIR"], "receiver",
                  {"samples": list(incoming.values()), "readback_matches": True,
                   "sampled_layers": len(incoming), "sampled_elements_per_layer": 64})
    return result


vLLMColocateWorkerExtension._update_weights = _audit_bucket
vLLMColocateWorkerExtension.update_weights_from_ipc = _audit_sync
# Alias retains the original extension's MRO and __new__ behavior.
DiagnosticWorkerExtension = vLLMColocateWorkerExtension
