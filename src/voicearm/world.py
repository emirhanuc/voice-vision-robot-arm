"""Sahne + robot + algı + ajanı tek yerden kurar (run.py, app.py, değerlendirme, testler)."""
from dataclasses import dataclass
from pathlib import Path

from .executor import Agent
from .nlp.parser import OllamaParser, RuleParser
from .perception import ColorDetector, Perception, SimCamera
from .sim.robot import ArmSim
from .sim.scene import Scene, build_scene


@dataclass
class World:
    scene: Scene
    robot: ArmSim
    perception: Perception
    agent: Agent


def build_world(robot="panda", gui=False, seed=None, detector=None, parser=None, llm=False,
                model="qwen2.5:7b", realtime=None, log=True, camera_png=None) -> World:
    scene = build_scene(gui=gui, seed=seed, robot=robot)
    arm = ArmSim(scene.robot_id, scene.spec, realtime=gui if realtime is None else realtime, log=log)
    lay = scene.spec.layout
    if detector is None or detector == "color":
        detector = ColorDetector()
    elif detector == "owl":
        from .perception import OwlDetector
        detector = OwlDetector()
    if camera_png:
        Path(camera_png).parent.mkdir(parents=True, exist_ok=True)
    perception = Perception(SimCamera(eye=lay.camera_eye, target=lay.camera_target), detector, arm,
                            save_path=camera_png)
    if parser is None:
        parser = OllamaParser(model=model) if llm else RuleParser()
    return World(scene, arm, perception, Agent(parser, perception, arm))
