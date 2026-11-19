import socket
import threading
import time

import pytest

from maia_infer import Broker, Client, Result, Worker


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Echo(Worker):
    service = "echo"

    def predict(self, request):
        return Result(arrays=request.arrays, meta={"echo": request.meta, "worker": id(self)})


class Stack:
    """A broker plus any number of workers, each in its own thread."""

    def __init__(self, worker_timeout=10.0, heartbeat=1.0):
        self.front = f"tcp://127.0.0.1:{free_port()}"
        self.back = f"tcp://127.0.0.1:{free_port()}"
        self.broker = Broker(self.front, self.back, heartbeat_interval=heartbeat, worker_timeout=worker_timeout)
        self.workers = []
        threading.Thread(target=self.broker.run, daemon=True).start()
        time.sleep(0.1)

    def add(self, cls=Echo, heartbeat_interval=0.2, **kw):
        worker = cls(broker=self.back, heartbeat_interval=heartbeat_interval, **kw)
        self.workers.append(worker)
        threading.Thread(target=worker.run, daemon=True).start()
        return worker

    def wait_for(self, service, n=1, timeout=3.0):
        end = time.time() + timeout
        while time.time() < end:
            if self.broker.stats()["services"].get(service, {}).get("workers", 0) >= n:
                return
            time.sleep(0.02)
        raise TimeoutError(f"{n} worker(s) for {service} never registered")

    def client(self, **kw):
        return Client(self.front, **kw)

    def close(self):
        for w in self.workers:
            w.stop()
        self.broker.stop()
        time.sleep(0.3)


@pytest.fixture
def stack():
    s = Stack()
    yield s
    s.close()
