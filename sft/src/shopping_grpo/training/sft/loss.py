"""Weighted causal CE on supervised rows, without full-context vocabulary logits."""


def weighted_linear_ce(hidden_states, lm_head, labels, weights, *, denominator=None,
                       use_liger=False, chunk_size=64):
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint

    if labels.shape != weights.shape or hidden_states.shape[:2] != labels.shape:
        raise ValueError('Hidden states, labels and weights must align')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Loss weights must be finite and nonnegative')
    if (weights[labels == -100] != 0).any():
        raise ValueError('Masked labels must have zero loss weight')
    shifted_labels = labels[:, 1:].reshape(-1)
    shifted_weights = weights[:, 1:].reshape(-1)
    valid = shifted_labels.ne(-100) & shifted_weights.gt(0)
    if not valid.any():
        raise ValueError('No supervised targets after the causal shift')
    # Accelerate may return FP32 hidden states outside the model's BF16 forward.
    # Restore the projection dtype before entering the fused CE's own AMP scope.
    states = hidden_states[:, :-1, :].reshape(-1, hidden_states.shape[-1])[valid]
    states = states.to(lm_head.weight.dtype).contiguous()
    targets = shifted_labels[valid].contiguous()
    token_weights = shifted_weights[valid].to(states.device)
    if use_liger:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        per_token = LigerFusedLinearCrossEntropyLoss(reduction='none')(
            lm_head.weight, states, targets, bias=lm_head.bias,
        )
    else:
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')

        def chunk_ce(state_chunk, target_chunk):
            logits = F.linear(state_chunk, lm_head.weight, lm_head.bias)
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            return F.cross_entropy(logits, target_chunk, reduction='none')

        pieces = []
        for start in range(0, len(targets), chunk_size):
            h, y = states[start:start + chunk_size], targets[start:start + chunk_size]
            pieces.append(checkpoint(chunk_ce, h, y, use_reentrant=False)
                          if torch.is_grad_enabled() else chunk_ce(h, y))
        per_token = torch.cat(pieces)
    divisor = token_weights.sum() if denominator is None else torch.as_tensor(
        denominator, device=states.device, dtype=token_weights.dtype,
    )
    if not torch.isfinite(divisor).all() or (divisor <= 0).any():
        raise ValueError('Weighted-token denominator must be positive and finite')
    return (per_token * token_weights).sum() / divisor, per_token, valid


def segment_trainer_class(trainer_base, *, use_liger):
    from collections import defaultdict
    import torch
    from shopping_grpo.training.sft.segments import SEGMENT_NAMES

    class SegmentTrainer(trainer_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.args.world_size != 1 or self.args.n_gpu > 1:
                raise ValueError('Segment loss currently requires a single GPU/process')
            # Trainer must not divide again by accumulation steps: compute_loss
            # uses the sum of weights across the entire accumulation window.
            self.model_accepts_loss_kwargs = True
            self._segment_stats = defaultdict(lambda: [0.0, 0])

        def _get_num_items_in_batch(self, batch_samples, device):
            if not batch_samples:
                return None
            return sum(batch['loss_weights'][..., 1:].sum() for batch in batch_samples).to(device)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            forwarded = dict(inputs)
            labels = forwarded.pop('labels')
            weights = forwarded.pop('loss_weights')
            segments = forwarded.pop('segment_ids')
            if not model.training:
                # Comparable validation NLL, irrespective of the training objective.
                weights = labels.ne(-100).to(weights.dtype)
                num_items_in_batch = None
            outputs = model(**forwarded, output_hidden_states=True, return_dict=True,
                            use_cache=False, logits_to_keep=1)
            loss, per_token, valid = weighted_linear_ce(
                outputs.hidden_states[-1], model.get_output_embeddings(), labels, weights,
                denominator=num_items_in_batch, use_liger=use_liger,
            )
            if model.training:
                selected_segments = segments[:, 1:].reshape(-1)[valid]
                detached = per_token.detach()
                for segment, name in SEGMENT_NAMES.items():
                    if segment == 0:
                        continue
                    selected = detached[selected_segments == segment]
                    self._segment_stats[name][0] += selected.sum().item()
                    self._segment_stats[name][1] += selected.numel()
            # Retain neither all hidden layers nor logits in Trainer's outputs.
            return (loss, {'loss': loss}) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            if 'loss' in logs and self._segment_stats:
                logs = dict(logs)
                total, count = 0.0, 0
                for name, (value, size) in self._segment_stats.items():
                    if size:
                        logs[name + '_nll'] = value / size
                    total += value
                    count += size
                if count:
                    logs['assistant_nll'] = total / count
                self._segment_stats.clear()
            return super().log(logs, *args, **kwargs)

    return SegmentTrainer
