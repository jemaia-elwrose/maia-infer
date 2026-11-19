"""Worker/broker lifecycle: giving up, surviving broker restarts, clean signals."""

import os
import signal
import subprocess
import sys
import threading
import time

from conftest import Echo, Stack, free_port

from maia_infer import Broker


def test_worker_gives_up_when_broker_never_appears():
    w = Echo(broker=f"tcp://127.0.0.1:{free_port()}", heartbeat_interval=0.1, broker_timeout=0.3, give_up_after=1.0)
    t0 = time.time()
    assert w.run() == 1
    assert time.time() - t0 < 3


def test_worker_reregisters_after_broker_restart():
    s = Stack()
    try:
        w = s.add()
        w.broker_timeout = 0.5
        s.wait_for("echo")
        s.broker.stop()
        time.sleep(1.5)  # worker notices and starts reconnecting
        b2 = Broker(s.front, s.back, heartbeat_interval=0.2)
        threading.Thread(target=b2.run, daemon=True).start()
        end = time.time() + 5
        while time.time() < end and b2.stats()["services"].get("echo", {}).get("workers", 0) < 1:
            time.sleep(0.05)
        assert b2.stats()["services"]["echo"]["workers"] == 1
        end = time.time() + 2  # the worker confirms once the new broker's first heartbeat arrives
        while time.time() < end and not (w._registered and w._lost_since is None):
            time.sleep(0.05)
        assert w._registered and w._lost_since is None
        b2.stop()
    finally:
        s.close()


def _spawn(*args):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    return subprocess.Popen(
        [sys.executable, "-m", "maia_infer", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def _exit_cleanly_on(sig):
    front, back = f"tcp://127.0.0.1:{free_port()}", f"tcp://127.0.0.1:{free_port()}"
    broker = _spawn("broker", "--frontend", front, "--backend", back)
    worker = _spawn("worker", "echo", "--broker", back)
    try:
        time.sleep(2)
        worker.send_signal(sig)
        assert worker.wait(timeout=5) == 0
        broker.send_signal(sig)
        assert broker.wait(timeout=5) == 0
        w_out, b_out = worker.stdout.read(), broker.stdout.read()
        assert "Traceback" not in w_out + b_out
        assert "worker 'echo' stopped" in w_out
        assert "disconnected" in b_out  # broker was told, not left to time out
        assert "broker stopped" in b_out
    finally:
        for proc in (worker, broker):
            if proc.poll() is None:
                proc.kill()


def test_ctrl_c_exits_cleanly():
    _exit_cleanly_on(signal.SIGINT)


def test_sigterm_exits_cleanly():  # what `docker stop` sends
    _exit_cleanly_on(signal.SIGTERM)


def test_cli_give_up_after_flag_is_honoured():
    """Regression: the flag was parsed but never passed to the worker."""
    proc = _spawn("worker", "echo", "--broker", f"tcp://127.0.0.1:{free_port()}", "--give-up-after", "1")
    # broker_timeout is 10 s by default, so the worker first notices the silence at ~10 s
    assert proc.wait(timeout=20) == 1
    assert "giving up" in proc.stdout.read()
