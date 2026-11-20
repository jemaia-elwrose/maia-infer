"""Transport overhead: maia-infer (ZeroMQ, raw buffers) vs HTTP.

The model is a no-op that returns a small detections array, so the numbers
measure only what it costs to move a frame to a model process and get a result
back. Three variants:

* http-json-base64 : Flask endpoint, frame as base64 inside JSON (the classic setup)
* http-raw         : Flask endpoint, frame as application/octet-stream
* maia-infer       : broker + worker over ZeroMQ, frame as a raw buffer
* maia-infer-ref   : the frame sits in a maia-ring ring; only a ~200-byte reference travels
                     and the worker reads the pixels in place (same machine)

Usage:  python bench/bench_transport.py --n 300 --size 1080p
"""

from __future__ import annotations

import argparse
import base64
import logging
import statistics
import threading
import time

import numpy as np
import requests
from maia_ring import RingWriter

from maia_infer import Broker, Client, Result, Worker

SIZES = {"480p": (480, 640, 3), "720p": (720, 1280, 3), "1080p": (1080, 1920, 3)}
DETS = np.random.rand(20, 6).astype(np.float32)  # 20 fake boxes [x1,y1,x2,y2,conf,cls]


def http_server(port: int) -> None:
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.post("/json")
    def json_endpoint():
        body = request.get_json()
        frame = np.frombuffer(base64.b64decode(body["image"]), np.uint8).reshape(body["shape"])
        assert frame.size
        return jsonify(detections=DETS.tolist())

    @app.post("/raw")
    def raw_endpoint():
        h, w, c = (int(x) for x in request.headers["X-Shape"].split(","))
        frame = np.frombuffer(request.get_data(), np.uint8).reshape(h, w, c)
        assert frame.size
        return jsonify(detections=DETS.tolist())

    app.run(port=port, threaded=True)


class NoOp(Worker):
    service = "noop"

    def predict(self, request):
        assert request.arrays[0].size
        return Result(arrays=[DETS])


def timed(fn, n: int) -> list[float]:
    for _ in range(10):
        fn()  # warm-up
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--size", choices=SIZES, default="1080p")
    args = ap.parse_args()
    frame = np.random.randint(0, 255, SIZES[args.size], dtype=np.uint8)

    threading.Thread(target=http_server, args=(18080,), daemon=True).start()
    threading.Thread(target=Broker("tcp://127.0.0.1:15555", "tcp://127.0.0.1:15556").run, daemon=True).start()
    threading.Thread(target=NoOp(broker="tcp://127.0.0.1:15556").run, daemon=True).start()
    time.sleep(1.5)

    session = requests.Session()
    client = Client("tcp://127.0.0.1:15555")

    def http_json():
        payload = {"image": base64.b64encode(frame.tobytes()).decode(), "shape": list(frame.shape)}
        np.asarray(session.post("http://127.0.0.1:18080/json", json=payload).json()["detections"], np.float32)

    def http_raw():
        r = session.post(
            "http://127.0.0.1:18080/raw",
            data=frame.tobytes(),
            headers={"Content-Type": "application/octet-stream", "X-Shape": ",".join(map(str, frame.shape))},
        )
        np.asarray(r.json()["detections"], np.float32)

    def zmq_call():
        client.infer("noop", frame)

    ring = RingWriter("maia-bench", slots=8, width=frame.shape[1], height=frame.shape[0])
    ref = ring.write(frame)  # written once, like a gateway decoding a camera

    def zmq_ref():
        client.infer("noop", ref)

    print(f"\nframe {args.size} {frame.shape}, {frame.nbytes / 1e6:.1f} MB, n={args.n}\n")
    print(f"{'transport':<18}{'p50 ms':>9}{'p95 ms':>9}{'req/s':>9}")
    variants = [("http-json-base64", http_json), ("http-raw", http_raw), ("maia-infer", zmq_call)]
    for name, fn in [*variants, ("maia-infer-ref", zmq_ref)]:
        lat = timed(fn, args.n)
        p50, p95 = statistics.median(lat), statistics.quantiles(lat, n=20)[18]
        print(f"{name:<18}{p50:>9.2f}{p95:>9.2f}{1000 / statistics.mean(lat):>9.0f}")
    ring.close()


if __name__ == "__main__":
    main()
