"""Uçtan uca senaryolar: ayrıştırma -> kamera -> kontrol -> kamerayla doğrulama.
Başarı ayrıca simülasyonun gerçek konumlarıyla kontrol edilir (robot bunları görmez)."""
import sys
from pathlib import Path

import pybullet as p
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.executor import Agent  # noqa: E402
from voicearm.nlp.parser import RuleParser  # noqa: E402
from voicearm.perception import ColorDetector, Perception, SimCamera  # noqa: E402
from voicearm.sim.robot import PandaSim  # noqa: E402
from voicearm.sim.scene import build_scene  # noqa: E402


@pytest.fixture
def world():
    scene = build_scene(gui=False)
    robot = PandaSim(scene.robot_id, log=False)
    agent = Agent(RuleParser(), Perception(SimCamera(), ColorDetector(), robot), robot)
    yield scene, agent
    p.disconnect()


def inside(scene, obj, tray, tol=0.07):
    a, b = scene.position(obj), scene.position(tray)
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def test_clarification_dialog(world):
    scene, agent = world
    out = agent.handle("küpü mavi kutuya koy")
    assert out[-1].startswith("?") and agent.pending
    out = agent.handle("kırmızı olanı")
    assert out[-1].startswith("✓"), out
    assert inside(scene, "kirmizi_kup", "mavi_kutu")


def test_hold_then_place_and_stack(world):
    scene, agent = world
    out = agent.handle("önce yeşil küpü kırmızı kutuya koy, sonra sarı küpü onun üstüne koy")
    assert all(o.startswith("✓") for o in out), out
    g, y = scene.position("yesil_kup"), scene.position("sari_kup")
    assert inside(scene, "yesil_kup", "kirmizi_kutu")
    assert abs(g[0] - y[0]) < 0.03 and abs(g[1] - y[1]) < 0.03 and y[2] > g[2] + 0.04   # üstünde


def test_errors_are_explained(world):
    _, agent = world
    assert agent.handle("mavi kutuya bırak")[-1].startswith("✗")      # elinde bir şey yok
    assert "görmedim" in agent.handle("turuncu küpü al")[-1]
