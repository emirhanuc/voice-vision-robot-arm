"""PyBullet sahnesi: masa, Franka Panda kolu, renkli küpler ve hedef kutular."""
import math
from dataclasses import dataclass, field

import pybullet as p
import pybullet_data

COLORS = {
    "kirmizi": [0.9, 0.1, 0.1, 1],
    "yesil": [0.1, 0.8, 0.2, 1],
    "mavi": [0.1, 0.3, 0.9, 1],
    "sari": [0.95, 0.85, 0.1, 1],
}

TABLE_Z = 0.625  # table.urdf üst yüzeyi
CUBE_HALF = 0.025


@dataclass
class SceneObject:
    name: str          # örn. "kirmizi_kup"
    color: str         # örn. "kirmizi"
    kind: str          # "kup" | "kutu"
    body_id: int


@dataclass
class Scene:
    client: int
    robot_id: int
    objects: dict = field(default_factory=dict)

    def position(self, name: str):
        pos, _ = p.getBasePositionAndOrientation(self.objects[name].body_id)
        return list(pos)

    def find(self, color: str | None = None, kind: str | None = None):
        return [o for o in self.objects.values()
                if (color is None or o.color == color) and (kind is None or o.kind == kind)]


def _make_cube(color, pos):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3, rgbaColor=COLORS[color])
    body = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=pos)
    p.changeDynamics(body, -1, lateralFriction=1.5)
    return body


def _make_tray(color, pos, size=0.07, wall=0.005, height=0.03):
    """Üstü açık kutu: taban + 4 duvar, tek statik gövde."""
    halfs, offsets = [], []
    halfs.append([size, size, wall]); offsets.append([0, 0, wall])
    for dx, dy, hx, hy in [(size, 0, wall, size), (-size, 0, wall, size),
                           (0, size, size, wall), (0, -size, size, wall)]:
        halfs.append([hx, hy, height / 2]); offsets.append([dx, dy, height / 2])
    rgba = [c * 0.7 for c in COLORS[color][:3]] + [1]
    col = p.createCollisionShapeArray([p.GEOM_BOX] * 5, halfExtents=halfs, collisionFramePositions=offsets)
    vis = p.createVisualShapeArray([p.GEOM_BOX] * 5, halfExtents=halfs, visualFramePositions=offsets,
                                   rgbaColors=[rgba] * 5)
    return p.createMultiBody(0, col, vis, basePosition=pos)


def _random_positions(seed, n, min_dist=0.09):
    """Çalışma alanında, kutulardan ve birbirinden uzak rastgele küp konumları."""
    import random
    rng = random.Random(seed)
    trays = [(0.35, 0.35), (0.35, -0.35)]
    out = []
    while len(out) < n:
        xy = (rng.uniform(0.38, 0.68), rng.uniform(-0.28, 0.28))
        if all(math.dist(xy, o) > min_dist for o in out) and all(math.dist(xy, t) > 0.14 for t in trays):
            out.append(xy)
    return out


def build_scene(gui: bool = True, seed: int | None = None) -> Scene:
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(1.2, 50, -35, [0.5, 0, TABLE_Z])

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.5, 0, 0], useFixedBase=True)
    robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, TABLE_Z], useFixedBase=True)

    scene = Scene(client=client, robot_id=robot)
    z = TABLE_Z + CUBE_HALF
    cube_xy = [("kirmizi", (0.45, -0.15)), ("yesil", (0.55, 0.0)),
               ("mavi", (0.45, 0.15)), ("sari", (0.6, -0.25))]
    if seed is not None:
        cube_xy = list(zip(COLORS, _random_positions(seed, len(COLORS))))
    for color, xy in cube_xy:
        name = f"{color}_kup"
        scene.objects[name] = SceneObject(name, color, "kup", _make_cube(color, [*xy, z]))
    for color, xy in [("mavi", (0.35, 0.35)), ("kirmizi", (0.35, -0.35))]:
        name = f"{color}_kutu"
        scene.objects[name] = SceneObject(name, color, "kutu", _make_tray(color, [*xy, TABLE_Z]))

    for _ in range(100):
        p.stepSimulation()
    return scene
