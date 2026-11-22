"""Rastgele sahnelerde algı + görev başarısı değerlendirmesi (README'deki metrikler).

    python scripts/evaluate.py --trials 20
    python scripts/evaluate.py --trials 10 --detector owl

Simülasyonun gerçek konumları SADECE burada, ölçüm için kullanılır; robot onları görmez.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.nlp.parser import RuleParser  # noqa: E402
from voicearm.perception import ColorDetector  # noqa: E402
from voicearm.sim.scene import COLORS  # noqa: E402
from voicearm.world import build_world  # noqa: E402

TR = {"kirmizi": "kırmızı", "yesil": "yeşil", "mavi": "mavi", "sari": "sarı"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--detector", choices=["color", "owl"], default="color")
    ap.add_argument("--robot", choices=["panda", "proto6"], default="panda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    detector = ColorDetector()
    if args.detector == "owl":
        from voicearm.perception import OwlDetector
        detector = OwlDetector()
    parser = RuleParser()

    loc_err, det_hits, det_total, success, fails = [], 0, 0, 0, []
    t0 = time.time()
    for seed in range(args.trials):
        w = build_world(robot=args.robot, seed=seed, detector=detector, parser=parser, log=False)
        scene, perception = w.scene, w.perception

        # 1) Algı doğruluğu: her nesne doğru renk/türle bulundu mu, konum hatası ne?
        dets = perception.observe()
        for obj in scene.objects.values():
            det_total += 1
            match = [d for d in dets if d.color == obj.color and d.kind == obj.kind]
            if len(match) == 1:
                det_hits += 1
                gt = np.array(scene.position(obj.name))
                loc_err.append(np.linalg.norm(match[0].position[:2] - gt[:2]) * 1000)

        # 2) Uçtan uca görev: rastgele küp -> rastgele kutu
        rng = random.Random(seed)
        c, t = rng.choice(list(COLORS)), rng.choice(["kirmizi", "mavi"])
        cmd = f"{TR[c]} küpü {TR[t]} kutuya koy"
        reply = w.agent.handle(cmd)[-1]
        gt_obj, gt_tray = scene.position(f"{c}_kup"), scene.position(f"{t}_kutu")
        ok = reply.startswith("✓") and max(abs(gt_obj[0] - gt_tray[0]), abs(gt_obj[1] - gt_tray[1])) < 0.07
        success += ok
        if not ok:
            fails.append((seed, cmd, reply))
        p.disconnect()
        print(f"[{seed + 1}/{args.trials}] {cmd}: {'✓' if not fails or fails[-1][0] != seed else '✗'}")

    loc_err = np.array(loc_err)
    res = {
        "robot": args.robot,
        "detector": args.detector,
        "trials": args.trials,
        "detection_rate": det_hits / det_total,
        "loc_err_mm_mean": float(loc_err.mean()),
        "loc_err_mm_p95": float(np.percentile(loc_err, 95)),
        "task_success_rate": success / args.trials,
        "failures": fails,
        "wall_time_s": round(time.time() - t0, 1),
    }
    out = Path(args.out or f"docs/eval_{args.robot}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "failures"}, indent=2))
    for f in fails:
        print("  ✗", f)


if __name__ == "__main__":
    main()
