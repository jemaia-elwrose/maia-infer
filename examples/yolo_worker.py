"""Ultralytics YOLO as a maia-infer worker (detection, micro-batched).

    maia-infer worker examples.yolo_worker:YoloWorker --max-batch 8 \
        --opt model=yolov8m.pt --opt conf=0.35 --opt device=0

Reply: one float32 array of shape (N, 6) = [x1, y1, x2, y2, conf, cls] per frame,
plus the class names in ``meta["names"]``. Per-request ``meta`` may override
``conf`` and filter ``classes``.
"""

from __future__ import annotations

import numpy as np

from maia_infer import Result, Worker


class YoloWorker(Worker):
    service = "yolo"
    max_batch = 8
    max_wait_ms = 4

    def setup(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(self.options.get("model", "yolov8n.pt"))
        self.conf = float(self.options.get("conf", 0.25))
        self.imgsz = int(self.options.get("imgsz", 640))
        self.device = self.options.get("device")  # None = auto
        self.half = str(self.options.get("half", "false")).lower() == "true"
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), verbose=False, device=self.device)

    def predict_batch(self, requests):
        frames = [r.arrays[0] for r in requests]
        conf = min(float(r.meta.get("conf", self.conf)) for r in requests)
        results = self.model.predict(
            frames, conf=conf, imgsz=self.imgsz, device=self.device, half=self.half, verbose=False
        )
        outputs = []
        for request, result in zip(requests, results, strict=True):
            boxes = result.boxes
            dets = np.concatenate(
                [boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None], boxes.cls.cpu().numpy()[:, None]],
                axis=1,
            ).astype(np.float32)
            # Apply per-request settings on top of the shared batch run.
            dets = dets[dets[:, 4] >= float(request.meta.get("conf", self.conf))]
            if classes := request.meta.get("classes"):
                dets = dets[np.isin(dets[:, 5], classes)]
            outputs.append(Result(arrays=[dets], meta={"names": self.names}))
        return outputs
