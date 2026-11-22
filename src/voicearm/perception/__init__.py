"""Algı katmanı: kamera -> dedektör -> robot koordinatlarında nesne listesi."""
from .camera import Camera, Frame, RealSenseCamera, SimCamera  # noqa: F401
from .detect import ColorDetector, Detection, OwlDetector, draw  # noqa: F401


class Perception:
    """Robotu kameranın görüş alanından çeker, görüntü alır, nesneleri tespit eder."""

    OBSERVE_POSE = [0.25, 0.0, 0.95]

    def __init__(self, camera: Camera, detector, robot=None, save_path=None):
        self.camera, self.detector, self.robot = camera, detector, robot
        self.save_path = save_path
        self.last_frame = None

    def observe(self) -> list[Detection]:
        if self.robot is not None:
            self.robot.move_to(self.OBSERVE_POSE)
        frame = self.camera.capture()
        # masadan 20 cm'den yüksekte olanlar (ör. gripper'da tutulan küp) sahnenin parçası değil
        dets = [d for d in self.detector.detect(frame) if d.height < 0.2]
        self.last_frame = frame
        if self.save_path:
            import cv2
            cv2.imwrite(str(self.save_path), draw(frame, dets))
        return dets
