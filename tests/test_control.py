import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.sim.control import CartesianLine, quintic  # noqa: E402


def test_quintic_boundaries():
    assert quintic(0) == (0, 0, 0)
    s, ds, dds = quintic(1)
    assert np.isclose(s, 1) and np.isclose(ds, 0) and np.isclose(dds, 0)


def test_line_respects_speed_limit():
    traj = CartesianLine.plan([0, 0, 0], [0.5, 0, 0], v_max=0.25)
    speeds = [np.linalg.norm(traj.sample(t)[1]) for t in np.linspace(0, traj.duration, 500)]
    assert max(speeds) <= 0.25 + 1e-6
    assert np.allclose(traj.sample(traj.duration)[0], [0.5, 0, 0])
