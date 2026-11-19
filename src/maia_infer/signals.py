"""Graceful shutdown on Ctrl+C (SIGINT) and `docker stop` / `kill` (SIGTERM).

Both signals are handled explicitly instead of relying on Python's default
KeyboardInterrupt, because a process started in the background by a
non-interactive shell inherits "ignore SIGINT" and would otherwise never stop.
"""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable

log = logging.getLogger("maia_infer")


def install_stop_handlers(stop: Callable[[], None]) -> None:
    if threading.current_thread() is not threading.main_thread():
        return  # signals can only be handled in the main thread (e.g. tests run components in threads)

    def handler(signum, _frame):
        log.info("received %s, shutting down", signal.Signals(signum).name)
        stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)
