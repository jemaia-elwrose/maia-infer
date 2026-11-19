"""Shared base for object-detection workers.

Every detector returns the same thing, so pipelines can swap models without
code changes: one ``float32`` array of shape ``(N, 6)`` per frame,
``[x1, y1, x2, y2, conf, class_id]`` in pixel coordinates of the input frame,
plus ``meta["names"]`` mapping class id -> name.

Per-request ``meta`` may contain:
    conf     minimum confidence (overrides the worker default)
    classes  list of class ids or class names to keep, e.g. ["person", "car"]
"""

from __future__ import annotations

import numpy as np

from ..protocol import Request
from ..worker import Result, Worker


def select_device(preference: str | None = "auto") -> str:
    """'auto' picks CUDA, then Apple Silicon (MPS), then CPU."""
    if preference and preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DetectionWorker(Worker):
    max_batch = 8
    max_wait_ms = 4

    #: Frames arrive as OpenCV BGR unless the client says otherwise via ``meta["rgb"]``.
    names: dict[int, str]

    def setup(self) -> None:
        self.device = select_device(self.options.get("device", "auto"))
        self.conf = float(self.options.get("conf", 0.25))
        self.load()
        self._name_to_id = {v: k for k, v in self.names.items()}
        h = w = int(self.options.get("warmup_size", 640))
        self.detect([np.zeros((h, w, 3), np.uint8)], self.conf)  # first call compiles kernels

    # ------------------------------------------------------------- subclass API
    def load(self) -> None:
        """Load the model into ``self.model`` and fill ``self.names``."""
        raise NotImplementedError

    def detect(self, frames_rgb: list[np.ndarray], conf: float) -> list[np.ndarray]:
        """Return one (N, 6) float32 array per RGB frame."""
        raise NotImplementedError

    # ---------------------------------------------------------------- plumbing
    def _class_filter(self, classes) -> np.ndarray | None:
        if not classes:
            return None
        ids = [self._name_to_id.get(c, -1) if isinstance(c, str) else int(c) for c in classes]
        return np.asarray(ids, np.float32)

    def predict_batch(self, requests: list[Request]) -> list[Result]:
        frames = []
        for r in requests:
            frame = r.arrays[0]
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"expected an HxWx3 image, got shape {frame.shape}")
            frames.append(frame if r.meta.get("rgb") else frame[:, :, ::-1])
        # One forward pass for the whole batch at the lowest requested threshold,
        # then apply each request's own settings.
        batch_conf = min(float(r.meta.get("conf", self.conf)) for r in requests)
        outputs = []
        for request, dets in zip(requests, self.detect(frames, batch_conf), strict=True):
            dets = dets[dets[:, 4] >= float(request.meta.get("conf", self.conf))]
            keep = self._class_filter(request.meta.get("classes"))
            if keep is not None:
                dets = dets[np.isin(dets[:, 5], keep)]
            outputs.append(Result(arrays=[np.ascontiguousarray(dets, np.float32)], meta={"names": self.names}))
        return outputs


def pack(xyxy, conf, cls) -> np.ndarray:
    xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
    return np.concatenate(
        [xyxy, np.asarray(conf, np.float32).reshape(-1, 1), np.asarray(cls, np.float32).reshape(-1, 1)], axis=1
    )
