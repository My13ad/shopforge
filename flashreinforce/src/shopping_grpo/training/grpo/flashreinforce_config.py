"""Typed veRL policy-loss options; imported only inside the training runtime."""
from dataclasses import dataclass

from verl.workers.config import PolicyLossConfig


@dataclass
class FlashPolicyLossConfig(PolicyLossConfig):
    loss_mode: str = "flashreinforce"
    sequence_kl_threshold: float = 1e-3
