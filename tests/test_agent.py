"""Uçtan uca senaryolar: ayrıştırma -> kamera -> kontrol -> kamerayla doğrulama.
Başarı ayrıca simülasyonun gerçek konumlarıyla kontrol edilir (robot bunları görmez)."""
import sys
from pathlib import Path

import pybullet as p
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.world import build_world  # noqa: E402


@pytest.fixture
def world():
    w = build_world(robot="panda", log=False)
    yield w.scene, w.agent
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


def test_four_cube_tower_stays_upright(world):
    scene, agent = world
    out = agent.handle("sarı küpü kırmızı küpün üstüne koy, sonra yeşil küpü sarı küpün üstüne koy, "
                       "sonra mavi küpü yeşil küpün üstüne koy")
    assert len(out) == 3 and all(o.startswith("✓") for o in out), out
    for _ in range(480):          # 2 sn bekle: kule devrilecekse devrilsin
        p.stepSimulation()
    base, top = scene.position("kirmizi_kup"), scene.position("mavi_kup")
    assert top[2] > base[2] + 0.14                                   # 4 kat yükseklik
    assert abs(top[0] - base[0]) < 0.015 and abs(top[1] - base[1]) < 0.015


def test_proto6_stepper_arm_pick_place_and_stack():
    """Kendi 6 eksenli kol: step motor modeli (konum kontrolü, tork/hız limiti, adım çözünürlüğü)."""
    w = build_world(robot="proto6", log=False)
    try:
        out = w.agent.handle("kırmızı küpü mavi kutuya koy, sonra yeşil küpü sarı küpün üstüne koy")
        assert all(o.startswith("✓") for o in out), out
        assert inside(w.scene, "kirmizi_kup", "mavi_kutu")
        g, y = w.scene.position("yesil_kup"), w.scene.position("sari_kup")
        assert abs(g[0] - y[0]) < 0.015 and abs(g[1] - y[1]) < 0.015 and g[2] > y[2] + 0.04
    finally:
        p.disconnect()
