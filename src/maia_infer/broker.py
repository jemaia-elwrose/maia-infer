"""Service-routing broker with worker heartbeats and credit-based dispatch.

Clients connect to the *frontend*, workers to the *backend*. Each worker
announces the service it serves (e.g. ``"yolo"``) and how many requests it can
hold at once (its *capacity*, which enables micro-batching). The broker:

* routes each request to a live worker of the requested service,
* never sends a worker more in-flight requests than its capacity,
* queues requests while all workers are busy and drops them once their
  deadline has passed (the client has given up by then anyway),
* evicts workers that stop sending heartbeats, so a crashed GPU process never
  becomes a black hole for requests,
* answers ``ping`` and ``stats`` itself, for health checks and dashboards.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import zmq

from . import protocol as p
from .signals import install_stop_handlers

log = logging.getLogger("maia_infer.broker")


@dataclass
class _Worker:
    identity: bytes
    service: str
    capacity: int
    credits: int
    last_seen: float
    served: int = 0


@dataclass
class _Pending:
    client: bytes
    frames: list
    request_id: str
    deadline: float


@dataclass
class _Service:
    name: str
    workers: deque = field(default_factory=deque)  # worker identities, round-robin order
    queue: deque = field(default_factory=deque)  # _Pending


class Broker:
    def __init__(
        self,
        frontend: str = "tcp://*:5555",
        backend: str = "tcp://*:5556",
        heartbeat_interval: float = 1.0,
        worker_timeout: float = 10.0,
        default_timeout_ms: int = 5000,
        max_queue: int = 1000,
        context: zmq.Context | None = None,
    ) -> None:
        self.frontend_addr = frontend
        self.backend_addr = backend
        self.heartbeat_interval = heartbeat_interval
        self.worker_timeout = worker_timeout
        self.default_timeout_ms = default_timeout_ms
        self.max_queue = max_queue
        self.ctx = context or zmq.Context.instance()
        self.workers: dict[bytes, _Worker] = {}
        self.services: dict[str, _Service] = {}
        self.started = time.time()
        self.counters = {"requests": 0, "replies": 0, "expired": 0, "rejected": 0, "evicted": 0}
        self._running = False

    # ------------------------------------------------------------------ setup
    def _bind(self) -> None:
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(self.frontend_addr)
        self.backend = self.ctx.socket(zmq.ROUTER)
        self.backend.setsockopt(zmq.LINGER, 0)
        self.backend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.backend.bind(self.backend_addr)
        log.info("broker listening: clients=%s workers=%s", self.frontend_addr, self.backend_addr)

    def stop(self) -> None:
        self._running = False

    # --------------------------------------------------------------- main loop
    def run(self) -> None:
        install_stop_handlers(self.stop)
        self._bind()
        poller = zmq.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        poller.register(self.backend, zmq.POLLIN)
        next_heartbeat = time.monotonic() + self.heartbeat_interval
        self._running = True
        try:
            while self._running:
                timeout_ms = max(0, int((next_heartbeat - time.monotonic()) * 1000))
                events = dict(poller.poll(min(timeout_ms, 100)))
                if self.backend in events:
                    self._on_worker(self.backend.recv_multipart(copy=False))
                if self.frontend in events:
                    self._on_client(self.frontend.recv_multipart(copy=False))
                now = time.monotonic()
                if now >= next_heartbeat:
                    self._heartbeat(now)
                    next_heartbeat = now + self.heartbeat_interval
                self._dispatch_all()
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self.frontend.close()
            self.backend.close()
            log.info("broker stopped")

    # ---------------------------------------------------------------- clients
    def _on_client(self, frames: list) -> None:
        client, body = frames[0].bytes, frames[1:]
        try:
            header = p.decode_header(body[0])
        except (p.ProtocolError, IndexError) as exc:
            self.counters["rejected"] += 1
            self._reply(client, p.Response.failure("", f"bad request: {exc}"))
            return

        request_id, op = str(header.get("id", "")), header.get("op", p.OP_INFER)
        if op == p.OP_PING:
            self._reply(client, p.Response(id=request_id, meta={"pong": True}))
            return
        if op == p.OP_STATS:
            self._reply(client, p.Response(id=request_id, meta=self.stats()))
            return

        self.counters["requests"] += 1
        name = str(header.get("service", ""))
        service = self.services.get(name)
        if service is None or not service.workers:
            self.counters["rejected"] += 1
            self._reply(client, p.Response.failure(request_id, self._no_workers(name)))
            return
        if len(service.queue) >= self.max_queue:
            self.counters["rejected"] += 1
            self._reply(client, p.Response.failure(request_id, f"service '{name}' queue full"))
            return
        timeout_ms = header.get("timeout_ms") or self.default_timeout_ms
        service.queue.append(_Pending(client, body, request_id, time.monotonic() + timeout_ms / 1000))

    def _reply(self, client: bytes, response: p.Response) -> None:
        self.frontend.send_multipart([client, *response.encode()])

    # ---------------------------------------------------------------- workers
    def _on_worker(self, frames: list) -> None:
        identity, command = frames[0].bytes, frames[1].bytes
        worker = self.workers.get(identity)
        now = time.monotonic()

        if command == p.READY:
            service_name = frames[2].bytes.decode()
            capacity = max(1, int(frames[3].bytes.decode())) if len(frames) > 3 else 1
            if worker is not None:  # re-registration after reconnect
                self._remove_worker(identity)
            worker = _Worker(identity, service_name, capacity, capacity, now)
            self.workers[identity] = worker
            self.services.setdefault(service_name, _Service(service_name)).workers.append(identity)
            log.info("worker %s ready for '%s' (capacity %d)", identity.hex()[:8], service_name, capacity)
            return

        if worker is None:
            # Unknown worker (e.g. broker restarted): ask it to re-register.
            self._send_worker(identity, [p.DISCONNECT])
            return

        worker.last_seen = now
        if command == p.REPLY:
            client = frames[2].bytes
            worker.credits = min(worker.capacity, worker.credits + 1)
            worker.served += 1
            self.counters["replies"] += 1
            self.frontend.send_multipart([client, *frames[3:]], copy=False)
        elif command == p.DISCONNECT:
            log.info("worker %s disconnected", identity.hex()[:8])
            self._remove_worker(identity)
        # HEARTBEAT: last_seen already refreshed.

    def _send_worker(self, identity: bytes, frames: list) -> bool:
        try:
            self.backend.send_multipart([identity, *frames], copy=False)
            return True
        except zmq.ZMQError:  # worker vanished (ROUTER_MANDATORY)
            self._remove_worker(identity)
            return False

    def _remove_worker(self, identity: bytes) -> None:
        worker = self.workers.pop(identity, None)
        if worker is None:
            return
        service = self.services.get(worker.service)
        if service is not None:
            try:
                service.workers.remove(identity)
            except ValueError:
                pass
            if not service.workers:
                # Nobody can serve the queue any more: fail fast.
                while service.queue:
                    item = service.queue.popleft()
                    self._reply(item.client, p.Response.failure(item.request_id, self._no_workers(service.name)))

    @staticmethod
    def _no_workers(name: str) -> str:
        return f"no workers for service '{name}'"

    # --------------------------------------------------------------- dispatch
    def _dispatch_all(self) -> None:
        now = time.monotonic()
        for service in self.services.values():
            while service.queue and service.queue[0].deadline < now:
                service.queue.popleft()
                self.counters["expired"] += 1
            while service.queue:
                worker = self._next_worker(service)
                if worker is None:
                    break
                item = service.queue.popleft()
                worker.credits -= 1
                if not self._send_worker(worker.identity, [p.REQUEST, item.client, *item.frames]):
                    if service.workers:
                        service.queue.appendleft(item)
                    else:
                        self._reply(item.client, p.Response.failure(item.request_id, self._no_workers(service.name)))

    def _next_worker(self, service: _Service) -> _Worker | None:
        for _ in range(len(service.workers)):
            identity = service.workers[0]
            service.workers.rotate(-1)
            worker = self.workers.get(identity)
            if worker is not None and worker.credits > 0:
                return worker
        return None

    def _heartbeat(self, now: float) -> None:
        for identity, worker in list(self.workers.items()):
            if now - worker.last_seen > self.worker_timeout:
                log.warning("worker %s ('%s') missed heartbeats, evicting", identity.hex()[:8], worker.service)
                self.counters["evicted"] += 1
                self._remove_worker(identity)
            else:
                self._send_worker(identity, [p.HEARTBEAT])

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "counters": dict(self.counters),
            "services": {
                name: {
                    "workers": len(s.workers),
                    "queued": len(s.queue),
                    "free_slots": sum(self.workers[w].credits for w in s.workers if w in self.workers),
                }
                for name, s in self.services.items()
            },
        }
