"""Frames sent by reference: the worker reads them in place from a maia-ring ring."""

import os
import threading
import time

import numpy as np
import pytest
from maia_ring import RingReader, RingWriter

from maia_infer import Result, StaleFrameError, Worker
from maia_infer import protocol as p


class Inspect(Worker):
    """Reports what it received instead of running a model."""

    service = "inspect"
    delay = 0.0

    def predict(self, request):
        if self.delay:
            time.sleep(self.delay)
        return Result(
            meta={
                "means": [float(a.mean()) for a in request.arrays],
                "writeable": [bool(a.flags.writeable) for a in request.arrays],
                "refs": len(request.frame_refs),
            }
        )


class SlowInspect(Inspect):
    service = "slow"
    delay = 0.3


class Unreachable(Inspect):
    """A worker on 'another machine': it cannot open any ring."""

    service = "remote"

    def setup(self):
        class NoRings:
            def get(self, ref, copy=False):
                raise FileNotFoundError(f"no shared memory named {ref['ring']}")

            def close(self):
                pass

        self._rings = NoRings()


@pytest.fixture
def ring(request):
    name = f"mi{os.getpid() % 10000}{abs(hash(request.node.name)) % 100000}"
    with RingWriter(name, slots=4, width=64, height=48) as w:
        yield w


def frame(value):
    return np.full((48, 64, 3), value, np.uint8)


def test_frame_by_reference_is_read_in_place(stack, ring):
    w = stack.add(Inspect)
    stack.wait_for("inspect")
    ref = ring.write(frame(7))
    with stack.client() as c:
        res = c.infer("inspect", ref)
    assert res.meta == {"means": [7.0], "writeable": [False], "refs": 1}  # a read-only view
    assert w.ref_stats["by_ref"] == 1


def test_frame_object_and_plain_arrays_mix_refs_first(stack, ring):
    stack.add(Inspect)
    stack.wait_for("inspect")
    ring.write(frame(3))
    with RingReader(ring.name) as r, stack.client() as c:
        f = r.latest(copy=False)
        res = c.infer("inspect", [np.full((2, 2), 9, np.uint8), f])
        del f
    assert res.meta["means"] == [3.0, 9.0]


def test_overwritten_before_the_worker_reads_it(stack, ring):
    stack.add(Inspect)
    stack.wait_for("inspect")
    ref = ring.write(frame(1))
    for i in range(4):
        ring.write(frame(10 + i))  # the ring laps the referenced frame
    with stack.client() as c, pytest.raises(StaleFrameError):
        c.infer("inspect", ref)


def test_overwritten_while_the_model_runs(stack, ring):
    w = stack.add(SlowInspect)
    stack.wait_for("slow")
    ref = ring.write(frame(1))

    def lap():
        time.sleep(0.1)  # the worker is inside predict() now
        for i in range(4):
            ring.write(frame(20 + i))

    t = threading.Thread(target=lap)
    t.start()
    with stack.client() as c, pytest.raises(StaleFrameError, match="overwritten while the model"):
        c.infer("slow", ref)
    t.join()
    assert w.ref_stats["stale"] == 1


def test_worker_without_access_gets_pixels_and_client_remembers(stack, ring):
    w = stack.add(Unreachable)
    stack.wait_for("remote")
    with stack.client() as c:
        first = c.infer("remote", ring.write(frame(5)))
        second = c.infer("remote", ring.write(frame(6)))
    assert first.meta["means"] == [5.0] and first.meta["refs"] == 0  # resent as pixels
    assert second.meta["means"] == [6.0] and second.meta["refs"] == 0
    assert w.ref_stats["unavailable"] == 1  # only the first request tried the reference
    assert "remote" in c._pixels_only


def test_protocol_carries_and_validates_frame_refs(ring):
    ref = ring.write(frame(1)).to_dict()
    req = p.Request.decode(p.Request(service="x", frame_refs=[ref]).encode())
    assert req.frame_refs == [ref] and req.arrays == []
    bad = p.Request(service="x").encode()
    bad[0] = bad[0].replace(b'"arrays"', b'"frame_refs": "nope", "arrays"')
    with pytest.raises(p.ProtocolError, match="frame_refs"):
        p.Request.decode(bad)
    failure = p.Response.decode(p.Response.failure("id", "gone", p.STALE_FRAME).encode())
    assert failure.error_code == p.STALE_FRAME
