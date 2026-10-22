"""Serialize environment RPCs across the single-host veRL worker processes.

ShopSimulator slots share one SQLite FTS5 connection. Concurrent use can change
rankings or yield invalid BM25 scores. A shared file lock preserves serial search
semantics while model generation remains concurrent. Environment files stay fixed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

from filelock import FileLock

from shopping_grpo.environment.client import ShopAgentEnv


class SerializedShopAgentEnv(ShopAgentEnv):
    """Use one inter-process request lock per backend URL on this host."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        identity = hashlib.sha256(self.base_url.encode("utf-8")).hexdigest()[:24]
        self._request_lock = FileLock(
            str(Path(tempfile.gettempdir()) / f"shopping-grpo-environment-{identity}.lock"),
            timeout=self.timeout,
        )

    def _send(self, payload):
        with self._request_lock:
            return super()._send(payload)
