"""Run a video or RTSP stream through the YOLO worker and draw the boxes.

python examples/video_client.py --source rtsp://camera/stream --show
"""

from __future__ import annotations

import argparse
import time

import cv2

from maia_infer import Client


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="file, RTSP URL or webcam index")
    ap.add_argument("--endpoint", default="tcp://localhost:5555")
    ap.add_argument("--service", default="yolo")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    client = Client(args.endpoint, timeout_ms=2000)
    frames, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = client.infer(args.service, frame, meta={"conf": 0.35})
        dets, names = res.arrays[0], res.meta.get("names", {})
        for x1, y1, x2, y2, conf, cls in dets:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (80, 200, 120), 2)
            label = f"{names.get(str(int(cls)), int(cls))} {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 120), 1)
        frames += 1
        if args.show:
            cv2.imshow("maia-infer", frame)
            if cv2.waitKey(1) == 27:
                break
        if frames % 100 == 0:
            print(f"{frames / (time.time() - t0):.1f} FPS, {len(dets)} detections")
    client.close()


if __name__ == "__main__":
    main()
