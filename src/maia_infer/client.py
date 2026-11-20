"""Synchronous client with timeouts and automatic retries.

    with Client("tcp://localhost:5555") as client:
        res = client.infer("yolo", [frame], meta={"conf": 0.4})
        boxes = res.arrays[0]

Frames from a maia-ring ring can be sent by reference, so the worker reads them in place
instead of receiving their pixels:

        frame = ring.latest(copy=False)          # or a FrameRef from the gateway
        res = client.infer("yolo", frame)

If the worker cannot open the ring (it runs on another machine), the client resends the
pixels and remembers to do so for that service. If the frame was overwritten before the
worker read it, ``StaleFrameError`` is raised: send a newer frame.

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


class StaleFrameError(InferenceError):
    """A frame sent by reference was overwritten before the worker could use it."""


def _as_ref(item: Any):
    """A maia-ring FrameRef for a Frame or FrameRef, None for anything else (duck-typed, so
    maia-ring is imported only when frames are actually used)."""
    if hasattr(item, "ref") and hasattr(item, "array"):  # maia_ring.Frame
        return item.ref
    if hasattr(item, "to_dict") and hasattr(item, "ring"):  # maia_ring.FrameRef
        return item
    return None


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
        self._pixels_only: set[str] = set()  # services whose workers cannot open our rings
        self._rings = None
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
        if self._rings is not None:
            self._rings.close()
            self._rings = None

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
        arrays: Any = None,
        meta: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> p.Response:
        """Run ``service`` on one input or a list of inputs. An input is a numpy array, or a
        maia-ring ``Frame`` / ``FrameRef`` sent by reference. Referenced frames reach the
        worker first in ``request.arrays``, followed by the plain arrays, in order."""
        items = [arrays] if arrays is not None and not isinstance(arrays, list) else list(arrays or [])
        refs = [r for r in (_as_ref(i) for i in items) if r is not None]
        pixels = [i for i in items if _as_ref(i) is None]
        if refs and service in self._pixels_only:
            pixels, refs = self._read_locally(refs) + pixels, []
        request = p.Request(service=service, arrays=pixels, meta=meta or {}, frame_refs=[r.to_dict() for r in refs])
        response = self.call(request, timeout_ms)
        if not response.ok and response.error_code == p.RING_UNAVAILABLE and refs:
            self._pixels_only.add(service)  # that worker is on another machine: stop trying
            request = p.Request(service=service, arrays=self._read_locally(refs) + pixels, meta=meta or {})
            response = self.call(request, timeout_ms)
        if not response.ok:
            if response.error_code == p.STALE_FRAME:
                raise StaleFrameError(response.error or "stale frame")
            raise InferenceError(response.error or "unknown error")
        return response

    def _read_locally(self, refs: list) -> list[np.ndarray]:
        from maia_ring import RingPool, StaleFrame

        if self._rings is None:
            self._rings = RingPool()
        try:
            return [self._rings.get(r, copy=True).array for r in refs]
        except StaleFrame as exc:
            raise StaleFrameError(str(exc)) from exc

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
