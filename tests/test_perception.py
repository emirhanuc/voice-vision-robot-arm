import sys
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.perception import ColorDetector, SimCamera  # noqa: E402
from voicearm.sim.scene import build_scene  # noqa: E402


def test_color_detector_finds_all_objects_within_1cm():
    scene = build_scene(gui=False, seed=0)
    try:
        dets = ColorDetector().detect(SimCamera().capture())
        for obj in scene.objects.values():
            match = [d for d in dets if d.color == obj.color and d.kind == obj.kind]
            assert len(match) == 1, obj.name
            err = np.linalg.norm(match[0].position[:2] - np.array(scene.position(obj.name))[:2])
            assert err < 0.01, (obj.name, err)
    finally:
        p.disconnect()
