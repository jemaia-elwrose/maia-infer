import threading
import time

import numpy as np
import pytest
from conftest import Stack

from maia_infer import InferenceError, InferenceTimeout, Worker


def test_echo_roundtrip(stack):
    stack.add()
    stack.wait_for("echo")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    with stack.client() as c:
        res = c.infer("echo", frame, meta={"cam": 7})
    np.testing.assert_array_equal(res.arrays[0], frame)
    assert res.meta["echo"] == {"cam": 7}


def test_ping_and_stats(stack):
    stack.add()
    stack.wait_for("echo")
    with stack.client() as c:
        assert c.ping()
        stats = c.stats()
    assert stats["services"]["echo"]["workers"] == 1


def test_unknown_service_fails_fast(stack):
    with stack.client(timeout_ms=2000) as c:
        t0 = time.time()
        with pytest.raises(InferenceError, match="no workers"):
            c.infer("nope", np.zeros(1, np.uint8))
        assert time.time() - t0 < 0.5


class Boom(Worker):
    service = "boom"

    def predict(self, request):
        if request.meta.get("explode"):
            raise ValueError("bad frame")
        return {"fine": True}


def test_worker_survives_exceptions(stack):
    stack.add(Boom)
    stack.wait_for("boom")
    with stack.client() as c:
        with pytest.raises(InferenceError, match="ValueError: bad frame"):
            c.infer("boom", meta={"explode": True})
        assert c.infer("boom").meta == {"fine": True}


def test_load_balances_across_workers(stack):
    stack.add()
    stack.add()
    stack.wait_for("echo", n=2)
    with stack.client() as c:
        seen = {c.infer("echo").meta["worker"] for _ in range(10)}
    assert len(seen) == 2


class Batcher(Worker):
    service = "batch"
    max_batch = 8
    max_wait_ms = 30

    def setup(self):
        self.sizes = []

    def predict_batch(self, requests):
        self.sizes.append(len(requests))
        time.sleep(0.02)  # simulate GPU work
        return [{"n": len(requests)} for _ in requests]


def test_micro_batching(stack):
    worker = stack.add(Batcher)
    stack.wait_for("batch")

    def fire():
        with stack.client() as c:
            for _ in range(4):
                c.infer("batch")

    threads = [threading.Thread(target=fire) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(worker.sizes) == 32
    assert max(worker.sizes) > 1  # requests were actually grouped


class Slow(Worker):
    service = "slow"

    def predict(self, request):
        time.sleep(0.5)
        return {}


def test_client_timeout(stack):
    stack.add(Slow)
    stack.wait_for("slow")
    with stack.client(timeout_ms=100, retries=1) as c:
        with pytest.raises(InferenceTimeout):
            c.infer("slow")


def test_dead_worker_is_evicted():
    s = Stack(worker_timeout=0.5, heartbeat=0.1)
    try:
        w = s.add()
        s.wait_for("echo")
        w.stop()  # sends DISCONNECT
        time.sleep(1.5)
        assert "echo" not in s.broker.stats()["services"] or s.broker.stats()["services"]["echo"]["workers"] == 0
        with s.client(timeout_ms=500) as c:
            with pytest.raises(InferenceError, match="no workers"):
                c.infer("echo")
    finally:
        s.close()


class Hung(Worker):
    """Registers, then stops heartbeating without saying goodbye (like a crash)."""

    service = "hung"

    def run(self):
        self.setup()
        self._connect()
        time.sleep(5)


def test_silent_worker_is_evicted_by_heartbeat_timeout():
    s = Stack(worker_timeout=0.5, heartbeat=0.1)
    try:
        s.add(Hung)
        s.wait_for("hung")
        time.sleep(1.2)
        assert s.broker.stats()["services"]["hung"]["workers"] == 0
    finally:
        s.close()


def test_idle_worker_stays_registered_with_default_timings():
    """Regression: broker heartbeats arriving as often as the worker's poll
    interval used to reset its wait forever, so the worker never sent its own
    heartbeat and was evicted every ~11 s. Uses the real defaults' ratio
    (broker beats faster than the worker polls), compressed in time."""
    s = Stack(worker_timeout=1.5, heartbeat=0.25)
    try:
        s.add(heartbeat_interval=0.5)
        s.wait_for("echo")
        time.sleep(4)  # > 2x worker_timeout
        stats = s.broker.stats()
        assert stats["counters"]["evicted"] == 0
        assert stats["services"]["echo"]["workers"] == 1
    finally:
        s.close()
