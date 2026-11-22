"""Desteklenen kollar ve her birinin çalışma alanı düzeni.

  panda  : Franka Emika Panda, 7 DOF, tork kontrollü (computed-torque)
  proto6 : Emirhan'ın 6 DOF prototip kolu, step motorlu (konum kontrolü + adım çözünürlüğü)
"""
import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Layout:
    """Masadaki yerleşim (robot tabanı masanın x=0 noktasında)."""
    cubes: list                 # varsayılan küp (x, y) konumları
    trays: dict                 # renk -> (x, y)
    random_x: tuple             # rastgele sahnede küp bölgesi
    random_y: tuple
    camera_eye: tuple
    camera_target: tuple
    observe_pose: list          # kamera bakarken uç noktanın konumu (görüşü kapatmasın)
    home_pose: list
    approach: float = 0.15      # nesnenin üstünden yaklaşma yüksekliği


@dataclass
class RobotSpec:
    name: str
    urdf: str
    arm_joints: list            # eklem adları, sırayla
    finger_joints: list
    ee_link: str
    rest: list                  # IK için tercih edilen duruş
    tau_max: list               # [Nm] eklem başına
    control: str                # "torque" | "stepper"
    finger_open: float
    finger_force: float
    layout: Layout
    kp: list = field(default_factory=list)
    kd: list = field(default_factory=list)
    vel_max: list = field(default_factory=list)       # [rad/s]
    steps_per_rad: list = field(default_factory=list)  # step motor çözünürlüğü
    pos_gain: list = field(default_factory=list)       # step sürücü konum döngüsü kazancı (eklem başına)
    v_max: float = 0.25         # Kartezyen hız limiti [m/s]
    down_euler: tuple = (math.pi, 0, 0)   # "aşağı bakan gripper" yönelimi (kolun sıfır duruşuna göre)
    pybullet_data: bool = True  # urdf, pybullet_data içinde mi


PANDA = RobotSpec(
    name="panda",
    urdf="franka_panda/panda.urdf",
    arm_joints=[f"panda_joint{i}" for i in range(1, 8)],
    finger_joints=["panda_finger_joint1", "panda_finger_joint2"],
    ee_link="panda_grasptarget",
    rest=[0, -0.4, 0, -2.4, 0, 2.0, 0.785],
    tau_max=[87, 87, 87, 87, 12, 12, 12],              # Franka datasheet
    control="torque",
    finger_open=0.04, finger_force=30,
    kp=[400, 400, 400, 400, 250, 150, 100],
    kd=[40, 40, 40, 40, 30, 25, 20],
    layout=Layout(
        cubes=[(0.45, -0.15), (0.55, 0.0), (0.45, 0.15), (0.6, -0.25)],
        trays={"mavi": (0.35, 0.35), "kirmizi": (0.35, -0.35)},
        random_x=(0.38, 0.68), random_y=(-0.28, 0.28),
        camera_eye=(1.15, 0.0, 1.45), camera_target=(0.45, 0.0, 0.62),
        observe_pose=[0.25, 0.0, 0.95], home_pose=[0.4, 0.0, 1.0],
    ),
)


def _stepper_res(gear, microstep=8, full_steps=200):
    return full_steps * microstep * gear / (2 * math.pi)


PROTO6 = RobotSpec(
    name="proto6",
    urdf=str(ROOT / "robots/proto6/proto6.urdf"),
    arm_joints=["j1", "j2", "j3", "j4", "j5", "j6"],
    finger_joints=["finger_left", "finger_right"],
    ee_link="grasp_target",
    rest=[0, 0.6, 1.4, 0, 1.1, 0],
    tau_max=[5.0, 20.0, 20.0, 1.5, 1.5, 0.5],          # redüktör çıkışı, tahmini (generate_urdf.py)
    control="stepper",
    finger_open=0.04, finger_force=20,
    vel_max=[2.5, 1.2, 1.2, 3.0, 3.0, 4.0],
    pos_gain=[0.3, 0.3, 0.3, 0.08, 0.08, 0.05],   # düşük ataletli bilek eklemleri yüksek kazançta salınıyordu
    steps_per_rad=[_stepper_res(20), _stepper_res(50), _stepper_res(50),
                   _stepper_res(5), _stepper_res(5), _stepper_res(1, microstep=16)],
    v_max=0.15,
    down_euler=(0, math.pi, 0),   # x etrafında 180° J6'yı sınıra (±175°) dayıyordu
    pybullet_data=False,
    layout=Layout(
        cubes=[(0.22, -0.10), (0.28, 0.0), (0.22, 0.10), (0.30, -0.14)],
        trays={"mavi": (0.12, 0.25), "kirmizi": (0.12, -0.25)},
        random_x=(0.18, 0.31), random_y=(-0.14, 0.14),
        camera_eye=(0.75, 0.0, 1.30), camera_target=(0.20, 0.0, 0.62),
        observe_pose=[0.08, 0.0, 0.88], home_pose=[0.15, 0.0, 0.88],
        approach=0.12,
    ),
)

ROBOTS = {"panda": PANDA, "proto6": PROTO6}
