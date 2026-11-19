"""Synchronous client with timeouts and automatic retries.

    with Client("tcp://localhost:5555") as client:
        res = client.infer("yolo", [frame], meta={"conf": 0.4})
        boxes = res.arrays[0]

If a reply does not arrive in time the socket is discarded and recreated (the
"lazy pirate" pattern), so a late reply from a slow or dead worker can never be
mistaken for the answer to a later request. A ``Client`` is not thread-safe;
use one per thread.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import zmq

from . import protocol as p


class InferenceError(RuntimeError):
    """The worker or broker reported an error."""


class InferenceTimeout(TimeoutError):
    """No reply after all retries."""


class Client:
    def __init__(
        self,
        endpoint: str = "tcp://localhost:5555",
        timeout_ms: int = 5000,
        retries: int = 2,
        context: zmq.Context | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.retries = retries
        self.ctx = context or zmq.Context.instance()
        self.socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.IDENTITY, uuid.uuid4().bytes)
        self.socket.connect(self.endpoint)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ calls
    def call(self, request: p.Request, timeout_ms: int | None = None, retries: int | None = None) -> p.Response:
        timeout_ms = timeout_ms or self.timeout_ms
        retries = self.retries if retries is None else retries
        request.timeout_ms = timeout_ms
        frames = request.encode()
        for _attempt in range(retries + 1):
            self.socket.send_multipart(frames, copy=False)
            while self.socket.poll(timeout_ms):
                response = p.Response.decode(self.socket.recv_multipart(copy=False))
                if response.id == request.id or response.id == "":
                    return response
                # Otherwise: a stale reply to an earlier, timed-out request. Ignore it.
            self._connect()
        raise InferenceTimeout(f"no reply from {self.endpoint} after {retries + 1} attempt(s) of {timeout_ms} ms")

    def infer(
        self,
        service: str,
        arrays: list[np.ndarray] | np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> p.Response:
        if isinstance(arrays, np.ndarray):
            arrays = [arrays]
        response = self.call(p.Request(service=service, arrays=list(arrays or []), meta=meta or {}), timeout_ms)
        if not response.ok:
            raise InferenceError(response.error or "unknown error")
        return response

    def ping(self, timeout_ms: int = 1000) -> bool:
        try:
            return self.call(p.Request(op=p.OP_PING), timeout_ms, retries=0).ok
        except InferenceTimeout:
            return False

    def stats(self, timeout_ms: int = 1000) -> dict[str, Any]:
        return self.call(p.Request(op=p.OP_STATS), timeout_ms, retries=0).meta


def detections(response: p.Response) -> list[dict[str, Any]]:
    """Turn a detection worker's reply into readable dicts.

    >>> for d in detections(client.infer("yolo", frame)):
    ...     print(d["name"], d["conf"], d["box"])
    """
    names = {int(k): v for k, v in response.meta.get("names", {}).items()}
    if not response.arrays:
        return []
    return [
        {
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "conf": float(conf),
            "class_id": int(cls),
            "name": names.get(int(cls), str(int(cls))),
        }
        for x1, y1, x2, y2, conf, cls in response.arrays[0]
    ]
