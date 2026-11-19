"""Ultralytics YOLO (v8/11/...) detection worker.

    maia-infer worker yolo --opt model=yolo11m.pt --opt device=auto --max-batch 8

Options: model (default yolo11n.pt), conf, imgsz (640), device (auto|cuda:0|mps|cpu), half (false).
Runs on NVIDIA GPUs, Apple Silicon (device=mps) and CPU.
"""

from __future__ import annotations

from .detection import DetectionWorker, pack


class YoloWorker(DetectionWorker):
    service = "yolo"

    def load(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(self.options.get("model", "yolo11n.pt"))
        self.imgsz = int(self.options.get("imgsz", 640))
        self.half = str(self.options.get("half", "false")).lower() == "true" and self.device.startswith("cuda")
        self.names = {int(k): v for k, v in self.model.names.items()}

    def detect(self, frames_rgb, conf):
        # Ultralytics expects BGR numpy arrays (OpenCV convention).
        frames_bgr = [f[:, :, ::-1] for f in frames_rgb]
        results = self.model.predict(
            frames_bgr, conf=conf, imgsz=self.imgsz, device=self.device, half=self.half, verbose=False
        )
        return [
            pack(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()) for r in results
        ]
