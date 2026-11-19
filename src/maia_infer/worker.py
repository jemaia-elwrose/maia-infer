"""Worker base class: subclass it, load a model in ``setup`` and implement
``predict`` (one request) or ``predict_batch`` (micro-batched requests).

    class Upper(Worker):
        service = "upper"

        def predict(self, request):
            return Result(meta={"text": request.meta["text"].upper()})

    Upper(broker="tcp://localhost:5556").run()

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
        self.ctx = context or zmq.Context.instance()
        self._running = False
        self.socket: zmq.Socket | None = None

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
        self._running = False

    def _connect(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.broker)
        self.socket.send_multipart([p.READY, self.service.encode(), str(self.max_batch).encode()])
        self._last_broker_msg = time.monotonic()
        log.info("worker '%s' registered with %s (batch=%d)", self.service, self.broker, self.max_batch)

    def run(self) -> None:
        self.setup()
        self._connect()
        self._running = True
        next_heartbeat = time.monotonic() + self.heartbeat_interval
        try:
            while self._running:
                batch = self._collect(timeout_ms=int(self.heartbeat_interval * 1000))
                if batch:
                    self._process(batch)
                now = time.monotonic()
                if now >= next_heartbeat:
                    self.socket.send_multipart([p.HEARTBEAT])
                    next_heartbeat = now + self.heartbeat_interval
                if now - self._last_broker_msg > self.broker_timeout:
                    log.warning("broker silent for %.0fs, reconnecting", self.broker_timeout)
                    self._connect()
        finally:
            try:
                self.socket.send_multipart([p.DISCONNECT], flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                pass
            self.socket.close(linger=0)
            self.teardown()

    def _collect(self, timeout_ms: int) -> list[tuple[bytes, list]]:
        """Wait for the first request, then keep collecting until the batch is
        full or ``max_wait_ms`` has elapsed."""
        batch: list[tuple[bytes, list]] = []
        deadline = None
        wait = timeout_ms
        while len(batch) < self.max_batch:
            if not self.socket.poll(wait):
                break
            frames = self.socket.recv_multipart(copy=False)
            self._last_broker_msg = time.monotonic()
            command = frames[0].bytes
            if command == p.REQUEST:
                batch.append((frames[1].bytes, frames[2:]))
                if deadline is None:
                    deadline = time.monotonic() + self.max_wait_ms / 1000
            elif command == p.DISCONNECT:
                self._connect()  # broker forgot us (e.g. restart): re-register
            if deadline is not None:
                wait = max(0, int((deadline - time.monotonic()) * 1000))
                if wait == 0 and not self.socket.poll(0):
                    break
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
        requests = [r for _, r in decoded]
        try:
            outputs = self.predict_batch(requests)
            if len(outputs) != len(requests):
                raise RuntimeError(f"predict_batch returned {len(outputs)} outputs for {len(requests)} requests")
            responses = [_to_response(r.id, o) for r, o in zip(requests, outputs, strict=True)]
        except Exception as exc:  # never let one bad input kill the worker
            log.exception("inference failed")
            responses = [p.Response.failure(r.id, f"{type(exc).__name__}: {exc}") for r in requests]
        for (client, _), response in zip(decoded, responses, strict=True):
            self._reply(client, response)
