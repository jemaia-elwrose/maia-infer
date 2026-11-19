"""RF-DETR (Roboflow, Apache-2.0) detection worker: a real-time DETR that needs no NMS.

    maia-infer worker rfdetr --opt size=medium --opt device=auto --max-batch 4

Options: size (nano|small|medium|base|large, default medium), weights (fine-tuned
checkpoint, optional), conf, device (auto|cuda|mps|cpu), optimize:
    eager    (default) optimized inference snapshot, any batch size
    compile  compiled for a fixed batch of --max-batch (CUDA graphs on GPU): fastest,
             slower startup; partial batches are padded
    none     plain model, as trained

Pretrained checkpoints predict COCO *category ids* (1-based, with gaps), unlike
Ultralytics' 0-79; ``meta["names"]`` always carries the right mapping, and
filtering by class name (``meta={"classes": ["person"]}``) works for both.
On Apple Silicon, set PYTORCH_ENABLE_MPS_FALLBACK=1 for ops MPS lacks.
"""

from __future__ import annotations

import logging

import numpy as np

from .detection import DetectionWorker, pack

log = logging.getLogger("maia_infer.rfdetr")

_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


class RFDetrWorker(DetectionWorker):
    service = "rfdetr"
    max_batch = 4

    def load(self) -> None:
        import rfdetr

        size = self.options.get("size", "medium").lower()
        if size not in _VARIANTS:
            raise ValueError(f"size must be one of {sorted(_VARIANTS)}")
        kwargs = {}
        if weights := self.options.get("weights"):
            kwargs["pretrain_weights"] = weights
        if not self.device.startswith("cuda"):
            kwargs["device"] = self.device  # let rfdetr auto-pick CUDA itself
        self.model = getattr(rfdetr, _VARIANTS[size])(**kwargs)
        self.names = self._id_to_name(list(self.model.class_names))

        self.optimize = self.options.get("optimize", "eager").lower()
        if self.optimize not in ("eager", "compile", "none"):
            raise ValueError("optimize must be eager, compile or none")
        self.predictor = None
        if self.optimize != "none":
            if hasattr(self.model, "inference"):  # rfdetr >= 1.12
                compiled = self.optimize == "compile"
                self.predictor = self.model.inference(compile=compiled, batch_size=self.max_batch if compiled else 1)
            else:
                log.warning("this rfdetr version has no optimized inference API; using the plain model")
                self.optimize = "none"

    @staticmethod
    def _id_to_name(class_names: list[str]) -> dict[int, str]:
        try:
            from rfdetr.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES
        except ImportError:  # rfdetr < 1.12
            try:
                from rfdetr.util.coco_classes import COCO_CLASSES

                COCO_CLASS_NAMES = [n for _, n in sorted(COCO_CLASSES.items())]
            except ImportError:
                return dict(enumerate(class_names))
        if class_names == list(COCO_CLASS_NAMES):
            return {int(k): v for k, v in COCO_CLASSES.items()}
        return dict(enumerate(class_names))

    def detect(self, frames_rgb, conf):
        frames = [np.ascontiguousarray(f) for f in frames_rgb]
        n = len(frames)
        if self.predictor is None:
            results = self.model.predict(frames, threshold=conf)
        else:
            if self.optimize == "compile" and n < self.max_batch:
                frames += [frames[-1]] * (self.max_batch - n)  # compiled graphs need the exact batch size
            results = self.predictor.predict(frames, threshold=conf, include_source_image=False)
        if not isinstance(results, list):
            results = [results]
        return [pack(d.xyxy, d.confidence, d.class_id) for d in results[:n]]
