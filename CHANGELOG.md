# Changelog

## 0.3.0
- Plugin workers: packages register workers under the `maia_infer.workers` entry-point
  group, so `pip install maia-face` makes `maia-infer worker face` available.
- README: licensing notes for the models you plug in (Ultralytics is AGPL-3.0).

## 0.2.1
- Broker logs show full worker ids (they were cut to 8 hex characters, so different
  workers looked identical).

## 0.2.0
- Built-in workers by name: `maia-infer worker yolo | rfdetr | echo`; custom workers from
  `file.py:Class` or `module:Class`.
- Shared `DetectionWorker` base: one output contract for every detector, class filters by
  name, BGR/RGB handling, automatic device choice (CUDA, Apple Silicon MPS, CPU).
- RF-DETR worker with optimized inference (`optimize=eager|compile|none`).
- `detections()` helper for readable results.
- Graceful shutdown on Ctrl+C and SIGTERM, `--give-up-after`, quieter reconnect logs.
- Fixed: idle workers re-registering every ~11 s (heartbeat starvation).
- GPU image on `python:3.12-slim` with pinned PyTorch wheels; headless OpenCV; build-time
  import check; persistent weight volumes.

## 0.1.0
- First release: broker with service routing, credit-based dispatch and heartbeats,
  micro-batching workers, retrying client, wire protocol without pickle, benchmark.
