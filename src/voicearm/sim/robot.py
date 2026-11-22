"""Robot arayüzü ve simülasyondaki kollar.

Simülasyon ve gerçek kol aynı arayüzü (RobotInterface) uygular; üst katmanlar farkı bilmez.

ArmSim iki kontrol mimarisini destekler (RobotSpec.control):
  "torque"  : computed-torque (Panda gibi tork kontrollü kollar)
  "stepper" : step motorlu kollar — sürücüye konum (adım) komutu gider; eklem hız/tork
              limitleri ve adım çözünürlüğü uygulanır, gereken tork izlenir (adım kaçırma riski).
"""
import math
import time
from abc import ABC, abstractmethod

import numpy as np
import pybullet as p

from .control import CartesianLine, ComputedTorqueController, Gains
from .robots import PANDA, RobotSpec


class RobotInterface(ABC):
    approach: float = 0.15

    @abstractmethod
    def move_to(self, pos, v_max: float | None = None): ...

    @abstractmethod
    def open_gripper(self): ...

    @abstractmethod
    def close_gripper(self) -> bool:
        """Kapatır; bir nesne kavrandıysa True döner."""

    @abstractmethod
    def home(self): ...

    def pick(self, pos) -> bool:
        x, y, z = pos
        a = self.approach
        self.open_gripper()
        self.move_to([x, y, z + a])
        self.move_to([x, y, z + 0.005], v_max=0.06)   # yaklaşma: yavaş, düz iniş
        grasped = self.close_gripper()
        self.move_to([x, y, z + a], v_max=0.10)
        return grasped

    def place(self, pos, drop_height=0.06):
        x, y, z = pos
        a = self.approach
        self.move_to([x, y, z + a])
        self.move_to([x, y, z + drop_height], v_max=0.06)
        self.open_gripper()
        self.move_to([x, y, z + a])


class ArmSim(RobotInterface):
    DT = 1 / 240

    def __init__(self, robot_id: int, spec: RobotSpec = PANDA, realtime: bool = False, log: bool = True):
        self.id, self.spec = robot_id, spec
        self.realtime = realtime
        self.approach = spec.layout.approach
        self.down = p.getQuaternionFromEuler(list(spec.down_euler))
        self.t = 0.0
        self.log = [] if log else None
        self.on_step = None   # her simülasyon adımında çağrılır (ör. video kaydı)
        self.finger_target = spec.finger_open
        self.torque_peak = np.zeros(len(spec.arm_joints))   # |tork| / limit, en yüksek
        self.saturation_time = np.zeros(len(spec.arm_joints))  # [s] motor tork limitinde geçen süre

        names = {p.getJointInfo(robot_id, i)[1].decode(): i for i in range(p.getNumJoints(robot_id))}
        links = {p.getJointInfo(robot_id, i)[12].decode(): i for i in range(p.getNumJoints(robot_id))}
        self.ARM = [names[n] for n in spec.arm_joints]
        self.FINGERS = [names[n] for n in spec.finger_joints]
        self.EE_LINK = links[spec.ee_link]
        self.MOVABLE = [i for i in range(p.getNumJoints(robot_id))
                        if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
        self.arm_idx = [self.MOVABLE.index(j) for j in self.ARM]     # IK/dinamik vektörlerindeki yerleri
        info = [p.getJointInfo(robot_id, j) for j in self.MOVABLE]
        self.lower = [i[8] for i in info]
        self.upper = [i[9] for i in info]
        self.n = len(self.ARM)
        self.tau_max = np.array(spec.tau_max, float)
        if spec.control == "torque":
            self.ctrl = ComputedTorqueController(Gains(np.array(spec.kp, float), np.array(spec.kd, float)),
                                                 self.tau_max)

        for j, q in zip(self.ARM, spec.rest):
            p.resetJointState(self.id, j, q)
        for f in self.FINGERS:
            p.resetJointState(self.id, f, spec.finger_open)
            p.changeDynamics(self.id, f, lateralFriction=2.0)
        # Parmakları mekanik olarak bağla (gerçek gripper'daki dişli/kayış bağı gibi): ikisi hep simetrik
        # hareket eder, nesne gripper'ın tam ortasına oturur. Bağ olmadan önce değen parmak
        # küpü ~14 mm itiyordu ve istiflenen kuleler kayıyordu.
        c = p.createConstraint(self.id, self.FINGERS[0], self.id, self.FINGERS[1], p.JOINT_GEAR,
                               [1, 0, 0], [0, 0, 0], [0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
        if spec.control == "torque":
            # Gerçek tork kontrolü için PyBullet'in dahili hız motorlarını kapat
            p.setJointMotorControlArray(self.id, self.ARM, p.VELOCITY_CONTROL, forces=[0] * self.n)
        self._q_hold = np.array(spec.rest, float)

    # ---------- durum ----------
    def _state(self):
        st = p.getJointStates(self.id, self.MOVABLE)
        return np.array([s[0] for s in st]), np.array([s[1] for s in st])

    def ee_position(self):
        return np.array(p.getLinkState(self.id, self.EE_LINK, computeForwardKinematics=True)[4])

    def _ik(self, pos):
        rest = [0.0] * len(self.MOVABLE)
        for k, q in zip(self.arm_idx, self.spec.rest):
            rest[k] = q
        q = p.calculateInverseKinematics(
            self.id, self.EE_LINK, list(pos), self.down,
            lowerLimits=self.lower, upperLimits=self.upper,
            jointRanges=[u - lo for u, lo in zip(self.upper, self.lower)],
            restPoses=rest, maxNumIterations=100, residualThreshold=1e-5)
        return np.array([q[k] for k in self.arm_idx])

    def _full(self, q_arm, q_all):
        out = list(q_all)
        for k, v in zip(self.arm_idx, q_arm):
            out[k] = float(v)
        return out

    def _jacobian(self, q_all):
        zeros = [0.0] * len(q_all)
        jl, ja = p.calculateJacobian(self.id, self.EE_LINK, [0, 0, 0], list(q_all), zeros, zeros)
        J = np.vstack([np.array(jl), np.array(ja)])
        return J[:, self.arm_idx]

    def _inverse_dynamics(self, q_all, qd_all):
        def f(q_arm, qd_arm, qdd_arm):
            q = self._full(q_arm, q_all)
            qd = self._full(qd_arm, qd_all)
            qdd = self._full(qdd_arm, [0.0] * len(q_all))
            tau = p.calculateInverseDynamics(self.id, q, qd, qdd)
            return [tau[k] for k in self.arm_idx]
        return f

    # ---------- kontrol döngüsü ----------
    def _control_step(self, q_des, qd_des, qdd_des, x_des=None):
        q_all, qd_all = self._state()
        q, qd = q_all[self.arm_idx], qd_all[self.arm_idx]
        if self.spec.control == "torque":
            tau, e = self.ctrl.torque(q, qd, q_des, qd_des, qdd_des, self._inverse_dynamics(q_all, qd_all))
            p.setJointMotorControlArray(self.id, self.ARM, p.TORQUE_CONTROL, forces=tau.tolist())
        else:
            # Step motor: komut adım çözünürlüğüne yuvarlanır, sürücü hız ve tork sınırında takip eder.
            spr = np.array(self.spec.steps_per_rad)
            q_cmd = np.round(q_des * spr) / spr
            gains = self.spec.pos_gain or [0.3] * self.n
            for j, qc, vmax, tmax, kp in zip(self.ARM, q_cmd, self.spec.vel_max, self.tau_max, gains):
                p.setJointMotorControl2(self.id, j, p.POSITION_CONTROL, targetPosition=float(qc),
                                        targetVelocity=0.0, force=float(tmax), maxVelocity=float(vmax),
                                        positionGain=float(kp), velocityGain=1.0)
            e = q_des - q
        if self.spec.control == "stepper":
            # Motorun bu adımda GERÇEKTEN uyguladığı tork. Limite dayandığı an = gerçek step motorda
            # adım kaçırma riski; bu süreler saturation_time'da birikir.
            tau = np.array([st[3] for st in p.getJointStates(self.id, self.ARM)])
            self.saturation_time += (np.abs(tau) >= 0.98 * self.tau_max) * self.DT
        self.torque_peak = np.maximum(self.torque_peak, np.abs(tau) / self.tau_max)
        p.setJointMotorControlArray(self.id, self.FINGERS, p.POSITION_CONTROL,
                                    targetPositions=[self.finger_target] * 2,
                                    forces=[self.spec.finger_force] * 2)
        p.stepSimulation()
        if self.on_step:
            self.on_step()
        self.t += self.DT
        if self.realtime:
            time.sleep(self.DT)
        if self.log is not None:
            self.log.append({
                "t": self.t,
                "x_des": None if x_des is None else np.asarray(x_des).tolist(),
                "x": self.ee_position().tolist(),
                "q_err": np.asarray(e).tolist(),
                "tau": np.asarray(tau).tolist(),
                "tau_max": self.tau_max.tolist(),
            })

    def hold(self, seconds):
        for _ in range(int(seconds / self.DT)):
            self._control_step(self._q_hold, np.zeros(self.n), np.zeros(self.n))

    def move_to(self, pos, v_max: float | None = None):
        v_max = min(v_max or self.spec.v_max, self.spec.v_max)
        traj = CartesianLine.plan(self.ee_position(), pos, v_max=v_max)
        qd_prev = None
        q_des = self._q_hold
        steps = int(math.ceil(traj.duration / self.DT))
        for k in range(1, steps + 1):
            x_d, v_d, _ = traj.sample(k * self.DT)
            q_des = self._ik(x_d)
            # İstenen eklem hızı: diferansiyel kinematik, q̇ = J⁺ [v; ω=0]
            q_all, _ = self._state()
            J = self._jacobian(self._full(q_des, q_all))
            qd_des = np.linalg.pinv(J) @ np.concatenate([v_d, np.zeros(3)])
            qdd_des = np.zeros(self.n) if qd_prev is None else (qd_des - qd_prev) / self.DT
            qd_prev = qd_des
            self._control_step(q_des, qd_des, qdd_des, x_des=x_d)
        self._q_hold = q_des
        self.hold(0.2)  # yerleşme süresi

    def home(self):
        self.move_to(self.spec.layout.home_pose)

    # ---------- gripper ----------
    def open_gripper(self):
        self.finger_target = self.spec.finger_open
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


def PandaSim(robot_id: int, realtime: bool = False, log: bool = True) -> ArmSim:
    """Geriye dönük uyumluluk."""
    return ArmSim(robot_id, PANDA, realtime, log)
