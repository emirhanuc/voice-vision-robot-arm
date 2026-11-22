"""Görüntüden nesne tespiti + 3D konum.

  ColorDetector : OpenCV HSV renk segmentasyonu. Hızlı, offline, GPU gerektirmez.
  OwlDetector   : OWLv2 açık sözlüklü tespit ("a red cube" gibi metin sorgusu).

İkisi de aynı Detection listesini döndürür. Nesne türü (küp / kutu) görüntüdeki piksel boyutundan
değil, nokta bulutundan ölçülen METRİK genişlikten belirlenir; kamera mesafesi değişse de çalışır.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Frame

# OpenCV HSV: H 0-179. Kırmızı iki uçta.
HSV_RANGES = {
    "kirmizi": [((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "sari": [((20, 120, 90), (35, 255, 255))],
    "yesil": [((40, 100, 60), (85, 255, 255))],
    "mavi": [((100, 120, 60), (130, 255, 255))],
}
COLOR_EN = {"kirmizi": "red", "yesil": "green", "mavi": "blue", "sari": "yellow"}
KIND_EN = {"kup": "cube", "kutu": "box"}

CUBE_MAX_WIDTH = 0.09   # m; bundan genişse "kutu" (tepsi)


@dataclass
class Detection:
    color: str
    kind: str              # "kup" | "kutu"
    position: np.ndarray   # nesnenin üst yüzey merkezi, robot tabanı koordinatları [m]
    width: float           # ölçülen genişlik [m]
    height: float          # masa üstünden yükseklik [m]
    pixel_box: tuple       # (x, y, w, h)
    score: float = 1.0

    @property
    def name(self):
        return f"{self.color}_{self.kind}"


def estimate_table_height(points: np.ndarray) -> float:
    z = points[..., 2]
    z = z[np.isfinite(z)]
    z = z[(z > 0.3) & (z < 1.2)]
    hist, edges = np.histogram(z, bins=200)
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


TOP_BAND = 0.004   # m; üst yüzey sayılacak kalınlık. Gerçek RealSense gürültüsünde ~0.008 yapın.


def measure(points: np.ndarray, mask: np.ndarray, table_z: float, top_band: float = TOP_BAND):
    """Maske içindeki 3D noktalardan üst yüzey merkezi, genişlik ve yükseklik.
    Üst yüzey dar bir bantla seçilir; geniş bant, kameraya bakan yan yüzün noktalarını da alıp
    merkezi kameraya doğru kaydırıyordu (~4 mm)."""
    pts = points[mask.astype(bool)]
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[pts[:, 2] > table_z + 0.003]
    if len(pts) < 20:
        return None
    top_z = np.percentile(pts[:, 2], 95)
    top = pts[pts[:, 2] > top_z - top_band]      # sadece üst yüzey (yan yüzleri at)
    lo, hi = np.percentile(pts[:, :2], 2, axis=0), np.percentile(pts[:, :2], 98, axis=0)
    width = float(np.max(hi - lo))
    if width < CUBE_MAX_WIDTH:
        xy = top[:, :2].mean(axis=0)              # dolu nesne: üst yüzeyin merkezi
    else:
        xy = (lo + hi) / 2                        # içi boş tepsi: kenarlar görünmeyebilir, sınır kutusu merkezi
    return np.array([*xy, top_z]), width, float(top_z - table_z)


class ColorDetector:
    def __init__(self, min_area=60):
        self.min_area = min_area

    def detect(self, frame: Frame) -> list[Detection]:
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        table_z = estimate_table_height(frame.points)
        out = []
        for color, ranges in HSV_RANGES.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                    continue
                m = measure(frame.points, labels == i, table_z)
                if m is None:
                    continue
                pos, width, height = m
                kind = "kup" if width < CUBE_MAX_WIDTH else "kutu"
                out.append(Detection(color, kind, pos, width, height, tuple(stats[i, :4])))
        return out


class OwlDetector:
    """OWLv2 açık sözlüklü dedektör. İlk çalıştırmada ~600 MB model iner.
    Kurulum: pip install torch transformers"""

    def __init__(self, model="google/owlv2-base-patch16-ensemble", threshold=0.2, device=None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Owlv2Processor.from_pretrained(model)
        self.model = Owlv2ForObjectDetection.from_pretrained(model).to(self.device).eval()
        self.threshold = threshold
        self.queries = [(c, k, f"a {COLOR_EN[c]} {KIND_EN[k]}") for c in COLOR_EN for k in KIND_EN]

    def detect(self, frame: Frame) -> list[Detection]:
        from PIL import Image
        img = Image.fromarray(frame.rgb)
        texts = [q[2] for q in self.queries]
        inputs = self.processor(text=[texts], images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        size = max(img.size)   # OWLv2 görüntüyü kare olarak dolgular
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=self.torch.tensor([[size, size]]))[0]

        table_z = estimate_table_height(frame.points)
        h, w = frame.rgb.shape[:2]
        out = []
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            mask = np.zeros((h, w), bool)
            mask[y0:y1, x0:x1] = True
            m = measure(frame.points, mask, table_z)
            if m is None:
                continue
            color, kind, _ = self.queries[int(label)]
            pos, width, height = m
            out.append(Detection(color, kind, pos, width, height, (x0, y0, x1 - x0, y1 - y0), float(score)))
        return nms_3d(out)


def nms_3d(dets: list[Detection], radius=0.04) -> list[Detection]:
    """Aynı yerdeki çakışan tespitlerden en yüksek skorluyu tut."""
    keep = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(np.linalg.norm(d.position[:2] - k.position[:2]) > radius for k in keep):
            keep.append(d)
    return keep


def draw(frame: Frame, dets: list[Detection]) -> np.ndarray:
    img = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR).copy()
    for d in dets:
        x, y, w, h = d.pixel_box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        label = f"{d.name} ({d.position[0]:.2f},{d.position[1]:.2f})"
        cv2.putText(img, label, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img
