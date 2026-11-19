import numpy as np
import pytest

from maia_infer.protocol import ProtocolError, Request, Response


def test_request_roundtrip():
    frame = np.random.randint(0, 255, (72, 128, 3), dtype=np.uint8)
    req = Request(service="yolo", arrays=[frame, np.ones(3, np.float32)], meta={"conf": 0.4}, timeout_ms=200)
    out = Request.decode([bytes(memoryview(f)) for f in req.encode()])
    assert out.service == "yolo" and out.meta == {"conf": 0.4} and out.timeout_ms == 200
    np.testing.assert_array_equal(out.arrays[0], frame)
    assert out.arrays[1].dtype == np.float32


def test_response_failure_roundtrip():
    out = Response.decode(Response.failure("abc", "boom").encode())
    assert not out.ok and out.error == "boom" and out.id == "abc"


def test_rejects_object_dtype():
    with pytest.raises(ProtocolError):
        Request(arrays=[np.array([object()])]).encode()


def test_rejects_size_mismatch():
    frames = Request(arrays=[np.zeros((4, 4), np.uint8)]).encode()
    with pytest.raises(ProtocolError):
        Request.decode([frames[0], b"\x00" * 3])


def test_rejects_garbage_header():
    with pytest.raises(ProtocolError):
        Request.decode([b"\x80\x04 not json"])


def test_rejects_unknown_version():
    with pytest.raises(ProtocolError):
        Request.decode([b'{"v": 99}'])
