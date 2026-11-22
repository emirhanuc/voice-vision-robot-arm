"""Yörünge planlama + computed-torque kontrol.

Gerçek bir kol kontrol yığınının sadeleştirilmiş hali:
  1) Planlayıcı: Kartezyen düz çizgi, quintic zaman profili (hız/ivme sınırda sıfır)
  2) Ters kinematik: her kontrol adımında istenen uç konumu -> istenen eklem açıları q_d
  3) Kontrolcü: computed-torque  tau = M(q)(qdd_d + Kp e + Kd edot) + C(q,qd)qd + g(q)
     Dinamik model PyBullet'ten (calculateInverseDynamics), tork gerçek Panda limitleriyle kırpılır.
"""
from dataclasses import dataclass, field

import numpy as np


def quintic(tau: float):
    """s(τ), ds/dτ, d²s/dτ² ; τ ∈ [0,1]. Başta/sonda hız ve ivme sıfır."""
    tau = min(max(tau, 0.0), 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


@dataclass
class CartesianLine:
    start: np.ndarray
    goal: np.ndarray
    duration: float

    @classmethod
    def plan(cls, start, goal, v_max=0.25, a_max=1.0, t_min=0.4):
        """Hız ve ivme limitlerine uyan en kısa süreyi seç (quintic için tepe hız 1.875·d/T)."""
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        d = float(np.linalg.norm(goal - start))
        T = max(t_min, 1.875 * d / v_max, np.sqrt(5.77 * d / a_max))
        return cls(start, goal, T)

    def sample(self, t):
        s, ds, dds = quintic(t / self.duration)
        delta = self.goal - self.start
        return (self.start + s * delta,
                ds / self.duration * delta,
                dds / self.duration**2 * delta)


@dataclass
class Gains:
    kp: np.ndarray = field(default_factory=lambda: np.array([400, 400, 400, 400, 250, 150, 100], float))
    kd: np.ndarray = field(default_factory=lambda: np.array([40, 40, 40, 40, 30, 25, 20], float))


# Franka Emika Panda eklem tork limitleri [Nm] (üretici datasheet'i)
TAU_MAX = np.array([87, 87, 87, 87, 12, 12, 12], float)


class ComputedTorqueController:
    def __init__(self, gains: Gains | None = None, tau_max=TAU_MAX):
        self.g = gains or Gains()
        self.tau_max = tau_max

    def torque(self, q, qd, q_des, qd_des, qdd_des, inverse_dynamics):
        e = q_des - q
        ed = qd_des - qd
        qdd_ref = qdd_des + self.g.kp * e + self.g.kd * ed
        tau = np.asarray(inverse_dynamics(q, qd, qdd_ref))
        return np.clip(tau, -self.tau_max, self.tau_max), e
