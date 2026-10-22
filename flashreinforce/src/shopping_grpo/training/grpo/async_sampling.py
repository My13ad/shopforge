"""Bounded completion-order rollout pool, independent of Ray and CUDA."""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from contextvars import ContextVar
from contextlib import contextmanager
from concurrent.futures import wait

policy_trace = ContextVar("shopping_async_policy_trace", default=None)


class GenerationGate:
    """Drain old-version generation before publishing weights; block new requests."""

    def __init__(self):
        import asyncio
        self.condition = asyncio.Condition()
        self.paused, self.active = False, 0

    async def enter(self, version):
        async with self.condition:
            await self.condition.wait_for(lambda: not self.paused)
            current = version()
            if not isinstance(current, int) or current < 0:
                raise ValueError("sampling weights have no published version")
            self.active += 1
            return current

    async def leave(self):
        async with self.condition:
            self.active -= 1
            self.condition.notify_all()

    async def pause(self, abort):
        async with self.condition:
            self.paused = True
        await abort()
        async with self.condition:
            await self.condition.wait_for(lambda: self.active == 0)

    async def resume(self, resume):
        await resume()
        async with self.condition:
            self.paused = False
            self.condition.notify_all()


@dataclass
class Submitted:
    future: Future
    cancel: object


class CompletionPool:
    """Keep a persistent rollout tail while a learner consumes completed items.

    submit(item) returns Submitted(future, cancel). The same interface accepts
    actual Ray ObjectRef futures and controlled CPU workers. No result filtering
    or trajectory reuse occurs here; callers form whole batches from this FIFO.
    """

    def __init__(self, items, submit, *, inflight, capacity, timeout=300):
        if min(inflight, capacity, timeout) <= 0:
            raise ValueError("completion pool bounds must be positive")
        self.items, self.submit = (items if hasattr(items, "try_next") else iter(items)), submit
        self.inflight, self.timeout = inflight, timeout
        self.ready, self.done = Queue(maxsize=capacity), Queue()
        self.stop, self.lock = Event(), Lock()
        self.dispatch_lock, self.paused = Lock(), Event()
        self.pending = {}
        self.closed = False
        self.submitted = self.finished = self.completed = self.delivered = 0
        self.thread = Thread(target=self._produce, name="flash-rollout-producer", daemon=True)
        self.thread.start()

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.ready.put(value, timeout=.05)
                return True
            except Full:
                pass
        return False

    def _on_done(self, future):
        # Count the worker's completion, including while the ready FIFO is full.
        # Enqueue timing alone cannot prove overlap with an optimizer update.
        with self.lock:
            self.finished += 1
        self.done.put(future)

    def _produce(self):
        exhausted = False
        try:
            while not self.stop.is_set():
                while not exhausted and len(self.pending) < self.inflight and not self.ready.full():
                    with self.dispatch_lock:
                        if self.paused.is_set() or self.stop.is_set():
                            break
                        try:
                            item = self.items.try_next() if hasattr(self.items, "try_next") else next(self.items)
                        except StopIteration:
                            exhausted = True
                            break
                        if item is None:
                            break  # Reserved source is temporarily unavailable; drain done work.
                        submitted = self.submit(item)
                        with self.lock:
                            self.pending[submitted.future] = (item, submitted.cancel)
                            self.submitted += 1
                        submitted.future.add_done_callback(self._on_done)
                if not self.pending:
                    if not exhausted:
                        self.stop.wait(.05)
                        continue
                    self._put(("end", None, None))
                    return
                try:
                    future = self.done.get(timeout=.05)
                except Empty:
                    continue
                item, _ = self.pending[future]
                output = future.result()
                if not self._put(("result", item, output)):
                    return
                with self.lock:
                    del self.pending[future]
                    self.completed += 1
        except BaseException as exc:
            self._put(("error", exc, None))

    def __iter__(self):
        return self

    def __next__(self):
        start = monotonic()
        try:
            kind, item, output = self.ready.get(timeout=self.timeout)
        except Empty as exc:
            raise TimeoutError("Flash rollout completion queue timed out") from exc
        if kind == "error":
            raise item
        if kind == "end":
            raise StopIteration
        self.delivered += 1
        return item, output, monotonic() - start

    def metrics(self):
        with self.lock:
            return {"flash_async/submitted": self.submitted,
                    "flash_async/finished": self.finished,
                    "flash_async/completed": self.completed,
                    "flash_async/delivered": self.delivered,
                    "flash_async/inflight": len(self.pending),
                    "flash_async/queued": self.ready.qsize()}

    @contextmanager
    def pause_dispatch(self):
        """Finish every submitted trajectory; retain the FIFO and resume afterward."""
        with self.dispatch_lock:
            if self.closed or self.paused.is_set():
                raise RuntimeError("cannot pause a closed or already paused rollout pool")
            self.paused.set()
        try:
            with self.lock:
                futures = list(self.pending)
            finished, unfinished = wait(futures, timeout=self.timeout)
            if unfinished:
                raise TimeoutError("training trajectories did not quiesce before validation")
            for future in finished:
                future.result()
            yield
        finally:
            self.paused.clear()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.thread.join(timeout=2)
        with self.lock:
            cancels = [(future, item, cancel) for future, (item, cancel) in self.pending.items()]
        for future, item, cancel in cancels:
            if future.done():
                continue
            if hasattr(self.items, "on_cancel"):
                self.items.on_cancel(item)
            cancel()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Flash rollout producer did not shut down")


async def resume_generation(generate, prompt_ids, sampling_params, *, make_output, trace=None):
    """Continue an interrupted assistant turn without replacing sampled tokens/logps."""
    import asyncio

    params = dict(sampling_params)
    maximum = int(params.get("max_tokens", params.get("max_new_tokens", 0)))
    if maximum <= 0:
        raise ValueError("partial generation requires an explicit positive token budget")
    params.pop("max_new_tokens", None)
    ids, logps, versions = [], [], []
    preempted = interruptions = 0
    while len(ids) < maximum:
        params["max_tokens"] = maximum - len(ids)
        output = await generate(prompt_ids=prompt_ids + ids, sampling_params=dict(params))
        count = len(output.token_ids)
        if count > params["max_tokens"] or (count and (output.log_probs is None or len(output.log_probs) != count)):
            raise ValueError("partial rollout lost actual sampler token probabilities")
        version = output.extra_fields.get("sampling_policy_version")
        if count and (not isinstance(version, int) or version < 0):
            raise ValueError("missing actual sampling policy version")
        ids.extend(output.token_ids)
        logps.extend(output.log_probs or [])
        versions.extend([version] * count)
        preempted += max(output.num_preempted or 0, 0)
        if output.stop_reason not in ("abort", "aborted"):
            break
        interruptions += 1
        await asyncio.sleep(0)
    if trace is not None:
        trace["versions"].extend(versions)
        trace["interruptions"] += interruptions
    extra = {"sampling_policy_versions": versions}
    if versions:
        extra.update(min_global_steps=min(versions), max_global_steps=max(versions), global_steps=max(versions))
    return make_output(token_ids=ids, log_probs=logps, num_preempted=preempted,
                       stop_reason=output.stop_reason, extra_fields=extra)
