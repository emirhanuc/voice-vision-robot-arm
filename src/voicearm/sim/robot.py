"""Robot arayüzü. Simülasyon ve gerçek kol aynı arayüzü uygular,
böylece üst katmanlar (görev yürütücü) hangisiyle konuştuğunu bilmez."""
import math
import time
from abc import ABC, abstractmethod

import numpy as np
import pybullet as p

from .control import CartesianLine, ComputedTorqueController


class RobotInterface(ABC):
    @abstractmethod
    def move_to(self, pos, v_max: float = 0.25): ...

    @abstractmethod
    def open_gripper(self): ...

    @abstractmethod
    def close_gripper(self) -> bool:
        """Kapatır; bir nesne kavrandıysa True döner."""

    def pick(self, pos, approach=0.15) -> bool:
        x, y, z = pos
        self.open_gripper()
        self.move_to([x, y, z + approach])
        self.move_to([x, y, z + 0.005], v_max=0.08)   # yaklaşma: yavaş, düz iniş
        grasped = self.close_gripper()
        self.move_to([x, y, z + approach], v_max=0.12)
        return grasped

    def place(self, pos, approach=0.15, drop_height=0.06):
        x, y, z = pos
        self.move_to([x, y, z + approach])
        self.move_to([x, y, z + drop_height], v_max=0.08)
        self.open_gripper()
        self.move_to([x, y, z + approach])

    def home(self):
        self.move_to([0.4, 0.0, 1.0])


class PandaSim(RobotInterface):
    ARM = list(range(7))
    FINGERS = [9, 10]
    MOVABLE = ARM + FINGERS            # PyBullet dinamik fonksiyonlarının beklediği sıra
    EE_LINK = 11
    REST = [0, -0.4, 0, -2.4, 0, 2.0, 0.785]
    DT = 1 / 240

    def __init__(self, robot_id: int, realtime: bool = False, log: bool = True):
        self.id = robot_id
        self.realtime = realtime
        self.down = p.getQuaternionFromEuler([math.pi, 0, 0])
        self.ctrl = ComputedTorqueController()
        self.t = 0.0
        self.log = [] if log else None
        self.finger_target = 0.04
        self.on_step = None   # her simülasyon adımında çağrılır (ör. video kaydı)

        for j, q in zip(self.ARM, self.REST):
            p.resetJointState(self.id, j, q)
        for f in self.FINGERS:
            p.resetJointState(self.id, f, 0.04)
            p.changeDynamics(self.id, f, lateralFriction=2.0)
        # Gerçek tork kontrolü için PyBullet'in dahili hız motorlarını kapat
        p.setJointMotorControlArray(self.id, self.ARM, p.VELOCITY_CONTROL, forces=[0] * 7)
        self._q_hold = np.array(self.REST, float)

    # ---------- durum ----------
    def _state(self):
        st = p.getJointStates(self.id, self.MOVABLE)
        q = np.array([s[0] for s in st])
        qd = np.array([s[1] for s in st])
        return q, qd

    def ee_position(self):
        return np.array(p.getLinkState(self.id, self.EE_LINK, computeForwardKinematics=True)[4])

    def _ik(self, pos):
        q = p.calculateInverseKinematics(
            self.id, self.EE_LINK, list(pos), self.down,
            lowerLimits=[-2.9, -1.8, -2.9, -3.0, -2.9, -0.1, -2.9, 0, 0],
            upperLimits=[2.9, 1.8, 2.9, 0.0, 2.9, 3.8, 2.9, 0.04, 0.04],
            jointRanges=[5.8, 3.6, 5.8, 3.0, 5.8, 3.9, 5.8, 0.04, 0.04],
            restPoses=self.REST + [0.02, 0.02], maxNumIterations=100, residualThreshold=1e-5)
        return np.array(q[:7])

    def _jacobian(self, q_all):
        zeros = [0.0] * len(q_all)
        jl, ja = p.calculateJacobian(self.id, self.EE_LINK, [0, 0, 0], list(q_all), zeros, zeros)
        return np.vstack([np.array(jl)[:, :7], np.array(ja)[:, :7]])

    def _inverse_dynamics(self, q_all, qd_all):
        def f(q_arm, qd_arm, qdd_arm):
            q = list(q_arm) + list(q_all[7:])
            qd = list(qd_arm) + list(qd_all[7:])
            qdd = list(qdd_arm) + [0.0, 0.0]
            return p.calculateInverseDynamics(self.id, q, qd, qdd)[:7]
        return f

    # ---------- kontrol döngüsü ----------
    def _control_step(self, q_des, qd_des, qdd_des, x_des=None):
        q, qd = self._state()
        tau, e = self.ctrl.torque(q[:7], qd[:7], q_des, qd_des, qdd_des,
                                  self._inverse_dynamics(q, qd))
        p.setJointMotorControlArray(self.id, self.ARM, p.TORQUE_CONTROL, forces=tau.tolist())
        # Parmaklar: ayrı, kuvvet sınırlı gripper kontrolcüsü (Franka Hand gibi)
        p.setJointMotorControlArray(self.id, self.FINGERS, p.POSITION_CONTROL,
                                    targetPositions=[self.finger_target] * 2, forces=[30, 30])
        p.stepSimulation()
        if self.on_step:
            self.on_step()
        self.t += self.DT
        if self.realtime:
            time.sleep(self.DT)
        if self.log is not None:
            x = self.ee_position()
            self.log.append({
                "t": self.t,
                "x_des": None if x_des is None else np.asarray(x_des).tolist(),
                "x": x.tolist(),
                "q_err": e.tolist(),
                "tau": tau.tolist(),
            })

    def hold(self, seconds):
        for _ in range(int(seconds / self.DT)):
            self._control_step(self._q_hold, np.zeros(7), np.zeros(7))

    def move_to(self, pos, v_max: float = 0.25):
        traj = CartesianLine.plan(self.ee_position(), pos, v_max=v_max)
        qd_prev = None
        n = int(math.ceil(traj.duration / self.DT))
        for k in range(1, n + 1):
            x_d, v_d, _ = traj.sample(k * self.DT)
            q_des = self._ik(x_d)
            # İstenen eklem hızı: diferansiyel kinematik, q̇ = J⁺ [v; ω=0]
            q_all, _ = self._state()
            J = self._jacobian(list(q_des) + list(q_all[7:]))
            qd_des = np.linalg.pinv(J) @ np.concatenate([v_d, np.zeros(3)])
            qdd_des = np.zeros(7) if qd_prev is None else (qd_des - qd_prev) / self.DT
            qd_prev = qd_des
            self._control_step(q_des, qd_des, qdd_des, x_des=x_d)
        self._q_hold = q_des
        self.hold(0.15)  # yerleşme süresi

    # ---------- gripper ----------
    def open_gripper(self):
        self.finger_target = 0.04
        self.hold(0.4)

    def finger_contact_force(self):
        """Her parmaktaki toplam normal temas kuvveti [N] (robotun kendisi hariç)."""
        forces = []
        for f in self.FINGERS:
            cps = [c for c in p.getContactPoints(bodyA=self.id, linkIndexA=f) if c[2] != self.id]
            forces.append(sum(c[9] for c in cps))
        return forces

    def close_gripper(self) -> bool:
        self.finger_target = 0.0
        self.hold(0.5)
        left, right = self.finger_contact_force()
        return left > 1.0 and right > 1.0   # iki parmakta da temas varsa kavrama başarılı
