"""Worker base class: subclass it, load a model in ``setup`` and implement
``predict`` (one request) or ``predict_batch`` (micro-batched requests).

    class Upper(Worker):
        service = "upper"

        def predict(self, request):
            return Result(meta={"text": request.meta["text"].upper()})

    Upper(broker="tcp://localhost:5556").run()

Requests may reference frames in a maia-ring shared-memory ring instead of carrying
their pixels. The worker resolves them into read-only arrays placed first in
``request.arrays`` (no copy), runs the model, and then checks the frames were not
overwritten meanwhile; a request whose frame changed under the model gets a
``stale_frame`` error instead of a result computed from mixed pixels. So workers must not
modify input arrays in place.

The worker keeps a heartbeat with the broker and reconnects automatically if
the broker restarts, so it can be deployed and scaled independently.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import zmq

from . import protocol as p
from .signals import install_stop_handlers

log = logging.getLogger("maia_infer.worker")


@dataclass
class Result:
    arrays: list[np.ndarray] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


Output = Result | np.ndarray | dict | None


def _to_response(request_id: str, output: Output) -> p.Response:
    if isinstance(output, p.Response):
        output.id = request_id
        return output
    if isinstance(output, Result):
        return p.Response(id=request_id, arrays=list(output.arrays), meta=dict(output.meta))
    if isinstance(output, np.ndarray):
        return p.Response(id=request_id, arrays=[output])
    if isinstance(output, dict):
        return p.Response(id=request_id, meta=output)
    if output is None:
        return p.Response(id=request_id)
    raise TypeError(f"unsupported worker output type: {type(output).__name__}")


class Worker:
    #: Service name clients use to reach this worker.
    service: str = "default"
    #: Maximum requests processed together. >1 enables micro-batching.
    max_batch: int = 1
    #: How long to wait for a batch to fill up before running it anyway.
    max_wait_ms: float = 5.0

    def __init__(
        self,
        broker: str = "tcp://localhost:5556",
        service: str | None = None,
        max_batch: int | None = None,
        max_wait_ms: float | None = None,
        heartbeat_interval: float = 1.0,
        broker_timeout: float = 10.0,
        give_up_after: float = 0.0,
        options: dict[str, Any] | None = None,
        context: zmq.Context | None = None,
    ) -> None:
        self.broker = broker
        #: Free-form settings (model path, device, thresholds), e.g. from ``--opt key=value``.
        self.options: dict[str, Any] = dict(options or {})
        if service is not None:
            self.service = service
        if max_batch is not None:
            self.max_batch = max_batch
        if max_wait_ms is not None:
            self.max_wait_ms = max_wait_ms
        self.heartbeat_interval = heartbeat_interval
        self.broker_timeout = broker_timeout
        #: Exit if the broker stays unreachable this long (seconds). 0 = keep trying forever,
        #: which is what you want under Docker/Kubernetes where the broker may restart.
        self.give_up_after = give_up_after
        self.ctx = context or zmq.Context.instance()
        self._running = False
        self._registered = False
        self._lost_since: float | None = None
        self.socket: zmq.Socket | None = None
        self._rings = None  # maia_ring.RingPool, created on the first frame reference
        #: Counters reported in logs: frames read by reference, stale, unavailable.
        self.ref_stats = {"by_ref": 0, "stale": 0, "unavailable": 0}

    # ------------------------------------------------------------ user hooks
    def setup(self) -> None:
        """Load models / warm up. Called once before registering with the broker."""

    def predict(self, request: p.Request) -> Output:
        raise NotImplementedError("implement predict() or predict_batch()")

    def predict_batch(self, requests: list[p.Request]) -> list[Output]:
        return [self.predict(r) for r in requests]

    def teardown(self) -> None:
        """Release resources on shutdown."""

    # --------------------------------------------------------------- runtime
    def stop(self) -> None:
        """Ask the worker to finish its current batch and exit (thread/signal safe)."""
        self._running = False

    def _connect(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.broker)
        self.socket.send_multipart([p.READY, self.service.encode(), str(self.max_batch).encode()])
        self._last_broker_msg = time.monotonic()
        self._registered = False

    def _on_broker_message(self) -> None:
        self._last_broker_msg = time.monotonic()
        if not self._registered:
            self._registered = True
            if self._lost_since is not None:
                log.info(
                    "broker is back after %.0fs; worker '%s' registered again",
                    time.monotonic() - self._lost_since,
                    self.service,
                )
                self._lost_since = None
            else:
                log.info("worker '%s' registered with %s (batch=%d)", self.service, self.broker, self.max_batch)

    def _install_signal_handlers(self) -> None:
        install_stop_handlers(self.stop)

    def run(self) -> int:
        """Serve until stopped. Returns 0 on a clean stop, 1 if the broker was given up on."""
        try:
            self.setup()  # may download/compile a model: plain Ctrl+C aborts it immediately
        except KeyboardInterrupt:
            log.info("interrupted during setup")
            return 130
        self._running = True
        self._install_signal_handlers()  # from here on, signals finish the current batch first
        self._connect()
        log.info("worker '%s' connecting to %s", self.service, self.broker)
        exit_code = 0
        next_heartbeat = time.monotonic() + self.heartbeat_interval
        next_status = 0.0
        try:
            while self._running:
                batch = self._collect(timeout_ms=int(self.heartbeat_interval * 1000))
                if batch:
                    self._process(batch)
                now = time.monotonic()
                if now >= next_heartbeat:
                    self.socket.send_multipart([p.HEARTBEAT])
                    next_heartbeat = now + self.heartbeat_interval

                if self._lost_since is not None and self.give_up_after and now - self._lost_since >= self.give_up_after:
                    log.error("broker %s unreachable for %.0fs, giving up", self.broker, now - self._lost_since)
                    exit_code = 1
                    break
                if now - self._last_broker_msg > self.broker_timeout:
                    if self._lost_since is None:
                        self._lost_since = self._last_broker_msg
                        hint = f", giving up after {self.give_up_after:.0f}s" if self.give_up_after else ""
                        log.warning("lost contact with broker %s, reconnecting%s", self.broker, hint)
                        next_status = now + 60
                    elif now >= next_status:
                        log.info("still waiting for broker %s (down %.0fs)", self.broker, now - self._lost_since)
                        next_status = now + 60
                    self._connect()
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            try:
                self.socket.send_multipart([p.DISCONNECT], flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                pass
            self.socket.close(linger=0)
            if self._rings is not None:
                self._rings.close()
            self.teardown()
            log.info("worker '%s' stopped", self.service)
        return exit_code

    def _collect(self, timeout_ms: int) -> list[tuple[bytes, list]]:
        """Wait up to ``timeout_ms`` for the first request, then keep collecting
        until the batch is full or ``max_wait_ms`` has elapsed.

        Both waits use absolute deadlines: control messages (broker heartbeats)
        must not restart the clock, or a worker that receives a heartbeat more
        often than ``timeout_ms`` would never return to send its own.
        """
        batch: list[tuple[bytes, list]] = []
        idle_deadline = time.monotonic() + timeout_ms / 1000
        batch_deadline: float | None = None
        while len(batch) < self.max_batch:
            deadline = batch_deadline if batch_deadline is not None else idle_deadline
            wait = max(0, int((deadline - time.monotonic()) * 1000))
            if not self.socket.poll(wait):
                break
            frames = self.socket.recv_multipart(copy=False)
            self._on_broker_message()
            command = frames[0].bytes
            if command == p.REQUEST:
                batch.append((frames[1].bytes, frames[2:]))
                if batch_deadline is None:
                    batch_deadline = time.monotonic() + self.max_wait_ms / 1000
            elif command == p.DISCONNECT:
                self._connect()  # broker forgot us (e.g. restart): re-register
        return batch

    def _reply(self, client: bytes, response: p.Response) -> None:
        self.socket.send_multipart([p.REPLY, client, *response.encode()], copy=False)

    def _process(self, batch: list[tuple[bytes, list]]) -> None:
        decoded: list[tuple[bytes, p.Request]] = []
        for client, frames in batch:
            try:
                decoded.append((client, p.Request.decode(frames)))
            except p.ProtocolError as exc:
                self._reply(client, p.Response.failure("", f"bad request: {exc}"))
        if not decoded:
            return
        held: dict[str, list] = {}  # request id -> ring frames to re-check after inference
        ready: list[tuple[bytes, p.Request]] = []
        for client, request in decoded:
            if request.frame_refs:
                failure = self._resolve_refs(request, held)
                if failure is not None:
                    self._reply(client, failure)
                    continue
            ready.append((client, request))
        if not ready:
            return
        requests = [r for _, r in ready]
        try:
            outputs = self.predict_batch(requests)
            if len(outputs) != len(requests):
                raise RuntimeError(f"predict_batch returned {len(outputs)} outputs for {len(requests)} requests")
            responses = [_to_response(r.id, o) for r, o in zip(requests, outputs, strict=True)]
        except Exception as exc:  # never let one bad input kill the worker
            log.exception("inference failed")
            responses = [p.Response.failure(r.id, f"{type(exc).__name__}: {exc}") for r in requests]
        for i, request in enumerate(requests):
            frames = held.get(request.id)
            if frames and not all(f.still_valid() for f in frames):
                self.ref_stats["stale"] += 1
                responses[i] = p.Response.failure(
                    request.id, "frame was overwritten while the model read it", p.STALE_FRAME
                )
            request.arrays = []  # drop the views into shared memory
        held.clear()
        for (client, _), response in zip(ready, responses, strict=True):
            self._reply(client, response)

    def _resolve_refs(self, request: p.Request, held: dict[str, list]) -> p.Response | None:
        """Replace frame references with read-only views into their rings."""
        try:
            from maia_ring import RingPool, StaleFrame
        except ImportError:  # pragma: no cover - maia-ring is a dependency
            return p.Response.failure(request.id, "maia-ring is not installed", p.RING_UNAVAILABLE)
        if self._rings is None:
            self._rings = RingPool()
        try:
            frames = [self._rings.get(ref, copy=False) for ref in request.frame_refs]
        except StaleFrame as exc:
            self.ref_stats["stale"] += 1
            return p.Response.failure(request.id, str(exc), p.STALE_FRAME)
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            self.ref_stats["unavailable"] += 1
            return p.Response.failure(request.id, f"cannot open ring: {exc}", p.RING_UNAVAILABLE)
        except (TypeError, KeyError) as exc:
            return p.Response.failure(request.id, f"bad frame reference: {exc}")
        self.ref_stats["by_ref"] += len(frames)
        held[request.id] = frames
        request.arrays = [f.array for f in frames] + request.arrays
        return None
