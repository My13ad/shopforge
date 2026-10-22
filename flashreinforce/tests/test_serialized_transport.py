"""Cross-process HTTP concurrency reaches the actual transport/file-lock seam."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import sys
import threading
import time

import pytest

from shopping_grpo.environment.client import ShopHttpError
from shopping_grpo.training.grpo.adapter.transport import SerializedShopAgentEnv


@contextmanager
def backend():
    counts = {"active": 0, "peak": 0, "calls": 0}
    counter_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with counter_lock:
                counts["active"] += 1
                counts["peak"] = max(counts["peak"], counts["active"])
                counts["calls"] += 1
            try:
                time.sleep(.08)
                data = json.dumps({"result": {"ok": True}}).encode()
                self.send_response(500 if body.get("fail") else 200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            finally:
                with counter_lock:
                    counts["active"] -= 1

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", counts
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_environment_requests_serialize_across_worker_processes():
    with backend() as (url, counts):
        program = (
            "import sys; from shopping_grpo.training.grpo.adapter.transport "
            "import SerializedShopAgentEnv; env=SerializedShopAgentEnv(base_url=sys.argv[1]); "
            "[env._call({'action':'probe'}) for _ in range(3)]"
        )
        workers = [subprocess.Popen([sys.executable, "-c", program, url],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                   for _ in range(4)]
        for worker in workers:
            _, error = worker.communicate(timeout=30)
            assert worker.returncode == 0, error.decode(errors="replace")
        assert counts["calls"] == 12
        assert counts["peak"] == 1


def test_failed_http_request_releases_the_process_lock():
    with backend() as (url, counts):
        first = SerializedShopAgentEnv(base_url=url, timeout=2)
        with pytest.raises(ShopHttpError):
            first._call({"fail": True})
        second = SerializedShopAgentEnv(base_url=url, timeout=2)
        assert second._call({}) == {"ok": True}
        assert counts["calls"] == 2
