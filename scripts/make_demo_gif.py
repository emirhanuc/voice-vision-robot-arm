"""README için demo GIF'i üretir: komutlar ekranda altyazı olarak görünür.

    python scripts/make_demo_gif.py [-o docs/demo.gif]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.executor import Agent  # noqa: E402
from voicearm.nlp.parser import RuleParser  # noqa: E402
from voicearm.perception import ColorDetector, Perception, SimCamera  # noqa: E402
from voicearm.sim.recorder import Recorder  # noqa: E402
from voicearm.sim.robot import PandaSim  # noqa: E402
from voicearm.sim.scene import build_scene  # noqa: E402

SCRIPT = [
    "küpü mavi kutuya koy",
    "kırmızı olanı",
    "önce yeşil küpü kırmızı kutuya koy, sonra sarı küpü onun üstüne koy",
]

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--out", default="docs/demo.gif")
args = ap.parse_args()

scene = build_scene(gui=False)
robot = PandaSim(scene.robot_id, log=False)
rec = Recorder()
robot.on_step = rec
agent = Agent(RuleParser(), Perception(SimCamera(), ColorDetector(), robot), robot)

for cmd in SCRIPT:
    rec.caption = f'Kullanıcı: "{cmd}"'
    for _ in range(20):
        rec.snap()
    replies = agent.handle(cmd)
    for reply in replies:
        print(cmd, "->", reply)
        rec.caption = "Robot: " + reply.lstrip("✓?✗ ").split(" (")[0]
        for _ in range(30):
            rec.snap()
Path(args.out).parent.mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)
rec.save_gif(args.out, fps=12, speedup=2)
import numpy as np
np.savez_compressed('logs/demo_frames.npz', frames=np.array(rec.frames))
print("GIF:", args.out, len(rec.frames), "kare")
