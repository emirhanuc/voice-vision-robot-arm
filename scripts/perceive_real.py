"""Gerçek RealSense kamerayla algı testi (robot gerekmez).

    python scripts/perceive_real.py --calib calib/T_base_cam.npy [--detector owl]
Kalibrasyon yoksa --calib vermeden çalıştır: konumlar kamera koordinatlarında olur.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.perception import ColorDetector, RealSenseCamera, draw  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--calib", default=None)
ap.add_argument("--detector", choices=["color", "owl"], default="color")
args = ap.parse_args()

T = np.load(args.calib) if args.calib else np.eye(4)
cam = RealSenseCamera(T)
if args.detector == "owl":
    from voicearm.perception import OwlDetector
    det = OwlDetector()
else:
    det = ColorDetector()
try:
    for _ in range(15):          # otomatik pozlamanın oturması için
        frame = cam.capture()
    dets = det.detect(frame)
    for d in dets:
        print(f"{d.name:14s} xyz={np.round(d.position, 3)}  genişlik={d.width*100:.1f} cm  skor={d.score:.2f}")
    Path("logs").mkdir(exist_ok=True)
    cv2.imwrite("logs/real.png", draw(frame, dets))
    print("Görüntü: logs/real.png")
finally:
    cam.close()
