"""Wire protocol shared by clients, the broker and workers.

A message is a ZeroMQ multipart message:

    [header, buffer_0, buffer_1, ...]

``header`` is a small UTF-8 JSON object; each buffer is the raw bytes of a
NumPy array whose dtype and shape are described in ``header["arrays"]``.

Why not pickle (``send_pyobj``)? Unpickling data from the network executes
arbitrary code. This format only ever decodes JSON and raw numeric buffers, so
a malicious peer can at worst send a malformed request. It is also faster:
image buffers are sent zero-copy instead of being serialised.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1

# Broker <-> worker control commands (single-byte frames).
READY = b"\x01"  # worker -> broker: [READY, service, capacity]
REQUEST = b"\x02"  # broker -> worker: [REQUEST, client_id, header, *buffers]
REPLY = b"\x03"  # worker -> broker: [REPLY, client_id, header, *buffers]
HEARTBEAT = b"\x04"  # both directions
DISCONNECT = b"\x05"  # worker -> broker on clean shutdown

# Request operations understood by the broker.
OP_INFER = "infer"
OP_PING = "ping"
OP_STATS = "stats"

_ALLOWED_DTYPES = {
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "bool",
}


class ProtocolError(ValueError):
    """Raised when a message cannot be decoded."""


def _describe(arr: np.ndarray) -> dict[str, Any]:
    if arr.dtype.name not in _ALLOWED_DTYPES:
        raise ProtocolError(f"unsupported dtype: {arr.dtype}")
    return {"dtype": arr.dtype.name, "shape": list(arr.shape)}


def _decode_arrays(specs: list[dict[str, Any]], buffers: list[bytes | memoryview]) -> list[np.ndarray]:
    if len(specs) != len(buffers):
        raise ProtocolError(f"header lists {len(specs)} arrays but {len(buffers)} buffers were sent")
    arrays = []
    for spec, buf in zip(specs, buffers, strict=True):
        dtype = spec.get("dtype")
        shape = tuple(int(s) for s in spec.get("shape", ()))
        if dtype not in _ALLOWED_DTYPES or any(s < 0 for s in shape):
            raise ProtocolError(f"bad array spec: {spec}")
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if len(buf) != expected:
            raise ProtocolError(f"buffer size {len(buf)} != expected {expected} for {spec}")
        arrays.append(np.frombuffer(buf, dtype=dtype).reshape(shape))
    return arrays


@dataclass
class Request:
    service: str = ""
    arrays: list[np.ndarray] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    op: str = OP_INFER
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timeout_ms: int | None = None

    def encode(self) -> list[bytes | np.ndarray]:
        header = {
            "v": PROTOCOL_VERSION,
            "id": self.id,
            "op": self.op,
            "service": self.service,
            "meta": self.meta,
            "arrays": [_describe(a) for a in self.arrays],
        }
        if self.timeout_ms is not None:
            header["timeout_ms"] = int(self.timeout_ms)
        return [json.dumps(header).encode()] + [np.ascontiguousarray(a) for a in self.arrays]

    @classmethod
    def decode(cls, frames: list[bytes | memoryview]) -> Request:
        header = decode_header(frames[0])
        return cls(
            service=str(header.get("service", "")),
            arrays=_decode_arrays(header.get("arrays", []), list(frames[1:])),
            meta=header.get("meta") or {},
            op=str(header.get("op", OP_INFER)),
            id=str(header.get("id", "")),
            timeout_ms=header.get("timeout_ms"),
        )


@dataclass
class Response:
    id: str
    ok: bool = True
    arrays: list[np.ndarray] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def encode(self) -> list[bytes | np.ndarray]:
        header = {
            "v": PROTOCOL_VERSION,
            "id": self.id,
            "ok": self.ok,
            "meta": self.meta,
            "arrays": [_describe(a) for a in self.arrays],
        }
        if self.error:
            header["error"] = self.error
        return [json.dumps(header).encode()] + [np.ascontiguousarray(a) for a in self.arrays]

    @classmethod
    def decode(cls, frames: list[bytes | memoryview]) -> Response:
        header = decode_header(frames[0])
        return cls(
            id=str(header.get("id", "")),
            ok=bool(header.get("ok", False)),
            arrays=_decode_arrays(header.get("arrays", []), list(frames[1:])),
            meta=header.get("meta") or {},
            error=header.get("error"),
        )

    @classmethod
    def failure(cls, request_id: str, message: str) -> Response:
        return cls(id=request_id, ok=False, error=message)


def decode_header(frame: bytes | memoryview) -> dict[str, Any]:
    try:
        header = json.loads(bytes(frame))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid header: {exc}") from exc
    if not isinstance(header, dict):
        raise ProtocolError("header must be a JSON object")
    if header.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {header.get('v')}")
    return header
