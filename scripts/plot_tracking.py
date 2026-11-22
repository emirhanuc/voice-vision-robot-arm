"""Kontrol kaydından takip hatası ve tork grafikleri üretir.

    python scripts/plot_tracking.py [logs/last_run.json] [-o docs/tracking.png]
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("log", nargs="?", default="logs/last_run.json")
ap.add_argument("-o", "--out", default="docs/tracking.png")
ap.add_argument("--title", default="Pick & place: uç noktası takibi ve eklem torkları")
args = ap.parse_args()

log = json.load(open(args.log))
t = np.array([d["t"] for d in log])
x = np.array([d["x"] for d in log])
tau = np.array([d["tau"] for d in log])
TAU_MAX = log[0].get("tau_max", [87, 87, 87, 87, 12, 12, 12])
mask = np.array([d["x_des"] is not None for d in log])
x_des = np.array([d["x_des"] if d["x_des"] else d["x"] for d in log])
err_mm = np.linalg.norm(x - x_des, axis=1) * 1000

fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for i, lbl in enumerate("xyz"):
    ax[0].plot(t, x_des[:, i], "--", lw=1, label=f"{lbl} istenen")
    ax[0].plot(t, x[:, i], lw=1.2, label=f"{lbl} gerçek")
ax[0].set_ylabel("Uç konumu [m]"); ax[0].legend(ncol=3, fontsize=8); ax[0].grid(alpha=.3)

ax[1].plot(t[mask], err_mm[mask], lw=1)
rms = np.sqrt((err_mm[mask] ** 2).mean())
ax[1].set_ylabel("Takip hatası [mm]")
ax[1].set_title(f"Kartezyen takip hatası  RMS={rms:.2f} mm  maks={err_mm[mask].max():.2f} mm", fontsize=10)
ax[1].grid(alpha=.3)

for j in range(tau.shape[1]):
    ax[2].plot(t, tau[:, j] / TAU_MAX[j] * 100, lw=1, label=f"J{j+1}")
ax[2].set_ylabel("Tork / limit [%]"); ax[2].set_xlabel("Zaman [s]")
ax[2].legend(ncol=tau.shape[1], fontsize=8); ax[2].grid(alpha=.3)

fig.suptitle(args.title, fontsize=12)
fig.tight_layout()
import pathlib; pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=130)
print(f"Kaydedildi: {args.out}  |  RMS {rms:.2f} mm")
