"""Ready-made workers, runnable by short name: ``maia-infer worker yolo``.

Heavy dependencies (torch, ultralytics, rfdetr) are imported only when a worker
is started, so the broker and client stay lightweight.
"""

BUILTIN = {
    "echo": "maia_infer.models.echo:EchoWorker",
    "yolo": "maia_infer.models.ultralytics_yolo:YoloWorker",
    "rfdetr": "maia_infer.models.rfdetr:RFDetrWorker",
}
