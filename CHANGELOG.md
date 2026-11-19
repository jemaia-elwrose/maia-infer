# Changelog

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
