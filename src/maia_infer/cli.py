"""Command line: ``maia-infer broker | worker | ping | stats``."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys


def _load_worker(spec: str):
    """Accepts a built-in name (``yolo``), ``package.module:Class`` or ``path/to/file.py:Class``."""
    from .models import BUILTIN

    spec = BUILTIN.get(spec, spec)
    target, _, attr = spec.rpartition(":")
    if not target or not attr:
        raise SystemExit(f"unknown worker '{spec}'. Use one of {sorted(BUILTIN)}, 'module:Class' or 'file.py:Class'")
    if target.endswith(".py"):
        path = os.path.abspath(target)
        if not os.path.isfile(path):
            raise SystemExit(f"worker file not found: {path}")
        module_spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
    else:
        sys.path.insert(0, os.getcwd())
        try:
            module = importlib.import_module(target)
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"cannot import '{target}' ({exc}). Run from the folder that contains it, or pass file.py:Class"
            )
    try:
        return getattr(module, attr)
    except AttributeError:
        raise SystemExit(f"'{target}' has no class '{attr}'")


def _parse_opts(items: list[str]) -> dict:
    opts = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--opt expects key=value, got '{item}'")
        opts[key] = value
    return opts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="maia-infer", description=__doc__)
    ap.add_argument("--log-level", default=os.getenv("MAIA_LOG_LEVEL", "INFO"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("broker", help="run the broker")
    b.add_argument("--frontend", default=os.getenv("MAIA_FRONTEND", "tcp://*:5555"))
    b.add_argument("--backend", default=os.getenv("MAIA_BACKEND", "tcp://*:5556"))
    b.add_argument("--worker-timeout", type=float, default=10.0)
    b.add_argument("--max-queue", type=int, default=1000)

    w = sub.add_parser("worker", help="run a worker: yolo | rfdetr | echo | module:Class | file.py:Class")
    w.add_argument("worker")
    w.add_argument("--broker", default=os.getenv("MAIA_BROKER", "tcp://localhost:5556"))
    w.add_argument("--service")
    w.add_argument("--max-batch", type=int)
    w.add_argument("--max-wait-ms", type=float)
    w.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE", help="worker option (repeatable)")

    for name in ("ping", "stats"):
        c = sub.add_parser(name, help=f"{name} the broker")
        c.add_argument("--endpoint", default=os.getenv("MAIA_ENDPOINT", "tcp://localhost:5555"))

    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "broker":
        from .broker import Broker

        Broker(args.frontend, args.backend, worker_timeout=args.worker_timeout, max_queue=args.max_queue).run()
    elif args.cmd == "worker":
        cls = _load_worker(args.worker)
        cls(
            broker=args.broker,
            service=args.service,
            max_batch=args.max_batch,
            max_wait_ms=args.max_wait_ms,
            options=_parse_opts(args.opt),
        ).run()
    else:
        from .client import Client

        with Client(args.endpoint) as client:
            if args.cmd == "ping":
                ok = client.ping()
                print("ok" if ok else "unreachable")
                raise SystemExit(0 if ok else 1)
            print(json.dumps(client.stats(), indent=2))


if __name__ == "__main__":
    main()
