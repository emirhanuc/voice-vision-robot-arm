"""RGB-D kamera arayüzü.

Her kamera aynı şeyi döndürür: renkli görüntü + robot taban koordinatlarında nokta bulutu
(her piksel için x,y,z). Böylece dedektörler simülasyonda da gerçek kamerada da aynı çalışır.

  SimCamera       : PyBullet sanal kamerası (sabit, masaya bakan "eye-to-hand" kurulum)
  RealSenseCamera : Intel RealSense D4xx + el-göz kalibrasyonundan gelen 4x4 dönüşüm
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray       # (H, W, 3) uint8
    points: np.ndarray    # (H, W, 3) float, robot tabanı koordinatları [m]; geçersiz = nan


class Camera(ABC):
    @abstractmethod
    def capture(self) -> Frame: ...


class SimCamera(Camera):
    def __init__(self, eye=(1.15, 0.0, 1.45), target=(0.45, 0.0, 0.62),
                 width=640, height=480, fov=55, near=0.1, far=3.0):
        import pybullet as p
        self.p = p
        self.w, self.h, self.near, self.far = width, height, near, far
        self.view = p.computeViewMatrix(eye, target, [0, 0, 1])
        self.proj = p.computeProjectionMatrixFOV(fov, width / height, near, far)
        # pikselden dünyaya geri izdüşüm için (PyBullet matrisleri sütun-öncelikli)
        V = np.array(self.view).reshape(4, 4, order="F")
        P = np.array(self.proj).reshape(4, 4, order="F")
        self.inv_pv = np.linalg.inv(P @ V)

    def capture(self) -> Frame:
        _, _, rgba, depth, _ = self.p.getCameraImage(
            self.w, self.h, self.view, self.proj, renderer=self.p.ER_TINY_RENDERER)
        rgb = np.reshape(rgba, (self.h, self.w, 4))[:, :, :3].astype(np.uint8)
        depth = np.reshape(depth, (self.h, self.w))

        xs = (np.arange(self.w) + 0.5) / self.w * 2 - 1
        ys = 1 - (np.arange(self.h) + 0.5) / self.h * 2
        X, Y = np.meshgrid(xs, ys)
        ndc = np.stack([X, Y, 2 * depth - 1, np.ones_like(X)], axis=-1).reshape(-1, 4)
        world = ndc @ self.inv_pv.T
        world = (world[:, :3] / world[:, 3:4]).reshape(self.h, self.w, 3)
        world[depth >= 0.9999] = np.nan   # gökyüzü / uzak düzlem
        return Frame(rgb, world)


class RealSenseCamera(Camera):
    """Gerçek donanım. T_base_cam: kamera -> robot tabanı 4x4 dönüşümü
    (el-göz kalibrasyonu ile bulunur, bkz. docs/calibration.md). Donanımsız test edilmedi."""

    def __init__(self, T_base_cam: np.ndarray, width=640, height=480, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        self.T = np.asarray(T_base_cam, float)
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)

    def capture(self) -> Frame:
        frames = self.align.process(self.pipe.wait_for_frames())
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        rgb = np.asanyarray(color.get_data()).copy()
        pc = self.rs.pointcloud()
        pc.map_to(color)
        verts = np.asanyarray(pc.calculate(depth).get_vertices()).view(np.float32).reshape(-1, 3)
        homo = np.c_[verts, np.ones(len(verts))] @ self.T.T
        pts = homo[:, :3].reshape(rgb.shape[0], rgb.shape[1], 3)
        pts[verts.reshape(rgb.shape[0], rgb.shape[1], 3)[:, :, 2] <= 0] = np.nan
        return Frame(rgb, pts)

    def close(self):
        self.pipe.stop()
