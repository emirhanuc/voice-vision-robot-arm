"""README için demo GIF'i üretir: komutlar ekranda altyazı olarak görünür.

    python scripts/make_demo_gif.py [-o docs/demo.gif]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.sim.recorder import Recorder  # noqa: E402
from voicearm.world import build_world  # noqa: E402

SCRIPT = [
    "küpü mavi kutuya koy",
    "kırmızı olanı",
    "önce yeşil küpü kırmızı kutuya koy, sonra sarı küpü onun üstüne koy",
]

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--out", default=None)
ap.add_argument("--robot", choices=["panda", "proto6"], default="panda")
args = ap.parse_args()

args.out = args.out or ("docs/demo.gif" if args.robot == "panda" else f"docs/demo_{args.robot}.gif")
w = build_world(robot=args.robot, log=False)
robot, agent = w.robot, w.agent
lay = w.scene.spec.layout
rec = Recorder() if args.robot == "panda" else Recorder(eye=(0.95, 0.35, 1.15), target=(0.15, 0.0, 0.68))
robot.on_step = rec

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
