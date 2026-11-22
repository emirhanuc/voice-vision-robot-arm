"""Emirhan'ın 6 eksenli prototip kolu (B seçeneği) için parametrik URDF üretici.

Özellikler: ~450 mm erişim (omuzdan flanşa), 1 kg yük, 3B baskı gövde,
step motorlar (J2-J3: NEMA23 + 50:1, J1: NEMA17 + 20:1), STM32F407 + DM542T sürücüler.

Geometri basit silindir/kutularla modellenir; kütle ve eylemsizlikler bu şekillerden
HESAPLANIR. CadQuery modelinden STL'ler geldiğinde `MESHES` sözlüğüne yollar yazılır,
görsel/çarpışma şekilleri gerçek parçalarla değişir; kinematik ve kütleler PARAMS'tan gelir.

    python robots/proto6/generate_urdf.py      # -> robots/proto6/proto6.urdf
"""
import math
from pathlib import Path

# ---------------------------------------------------------------- parametreler
# Uzunluklar [m], kütleler [kg] (motor + redüktör + baskı gövde tahmini). TAHMİNİ değerler:
# parçalar tartıldıkça/CAD'den okundukça güncelle.
PARAMS = {
    "base":     {"r": 0.070, "h": 0.060, "mass": 1.50},   # taban + J1 motoru (NEMA17 + EG17-G20)
    "shoulder": {"r": 0.055, "h": 0.100, "mass": 2.20},   # J2 + J3 motorları (2x NEMA23 + 50:1), omuzda
    "upper":    {"w": 0.060, "l": 0.200, "mass": 0.70},   # üst kol (J2 -> J3)
    "fore":     {"r": 0.032, "l": 0.180, "mass": 0.65},   # ön kol (J3 -> J5), J4 rulman + NEMA17 içinde
    "wrist":    {"r": 0.028, "l": 0.050, "mass": 0.30},   # J5 -> J6
    "flange":   {"r": 0.025, "l": 0.020, "mass": 0.10},   # ISO 9409 tarzı flanş
    "gripper":  {"palm": [0.050, 0.100, 0.030], "mass": 0.25,
                 "finger": [0.012, 0.015, 0.055], "finger_mass": 0.03, "stroke": 0.04},
}
J4_OFFSET = 0.10   # J4 (ön kol yuvarlanma) ekseninin J3'ten uzaklığı

# Eklem limitleri: [alt, üst] rad, maks tork [Nm] (redüktör çıkışı), maks hız [rad/s]
# Tork: motor tutma torku x oran x ~0.8 verim, redüktör izin verilen torkuyla sınırlı (tahmini).
# Hız: step motorun ~600 rpm üstünde tork düşer -> çıkış hızı = 600 rpm / oran.
JOINTS = {
    "j1": {"lim": (-2.97, 2.97), "tau": 5.0,  "vel": 2.5},   # NEMA17 + 20:1
    "j2": {"lim": (-1.75, 1.75), "tau": 20.0, "vel": 1.2},   # NEMA23 + 50:1
    "j3": {"lim": (-2.50, 2.50), "tau": 20.0, "vel": 1.2},   # NEMA23 + 50:1
    "j4": {"lim": (-3.05, 3.05), "tau": 1.5,  "vel": 3.0},   # NEMA17 + 5:1 (varsayım)
    "j5": {"lim": (-2.10, 2.10), "tau": 1.5,  "vel": 3.0},   # NEMA17 + 5:1 (varsayım)
    "j6": {"lim": (-3.05, 3.05), "tau": 0.5,  "vel": 4.0},   # NEMA14 (varsayım)
}

MESHES: dict[str, str] = {}   # örn. {"upper": "meshes/upper_arm.stl"} — CadQuery'den export

COLORS = {"orange": "0.95 0.45 0.1 1", "dark": "0.15 0.15 0.17 1", "grey": "0.6 0.6 0.62 1"}


# ---------------------------------------------------------------- yardımcılar
def cyl_inertia(m, r, h):
    ixx = m * (3 * r * r + h * h) / 12
    return ixx, ixx, m * r * r / 2


def box_inertia(m, x, y, z):
    return m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12


def geom(kind, dims, name):
    if name in MESHES:
        return f'<mesh filename="{MESHES[name]}"/>'
    if kind == "cyl":
        return f'<cylinder radius="{dims[0]}" length="{dims[1]}"/>'
    return f'<box size="{dims[0]} {dims[1]} {dims[2]}"/>'


def link(name, kind, dims, mass, xyz, color):
    if kind == "cyl":
        ixx, iyy, izz = cyl_inertia(mass, *dims)
    else:
        ixx, iyy, izz = box_inertia(mass, *dims)
    g = geom(kind, dims, name)
    o = f'<origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="0 0 0"/>'
    return f"""
  <link name="{name}">
    <inertial>{o}<mass value="{mass}"/>
      <inertia ixx="{ixx:.6f}" iyy="{iyy:.6f}" izz="{izz:.6f}" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual>{o}<geometry>{g}</geometry><material name="{color}"/></visual>
    <collision>{o}<geometry>{g}</geometry></collision>
  </link>"""


def joint(name, parent, child, xyz, axis, jtype="revolute"):
    j = JOINTS.get(name)
    limit = (f'<limit lower="{j["lim"][0]}" upper="{j["lim"][1]}" effort="{j["tau"]}" velocity="{j["vel"]}"/>'
             if j else "")
    return f"""
  <joint name="{name}" type="{jtype}">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="0 0 0"/>
    <axis xyz="{axis}"/>{limit}<dynamics damping="0.5" friction="0.05"/>
  </joint>"""


def build() -> str:
    P = PARAMS
    b, s, u, f, w, fl, g = (P[k] for k in ("base", "shoulder", "upper", "fore", "wrist", "flange", "gripper"))
    palm, fin = g["palm"], g["finger"]
    parts = ['<?xml version="1.0"?>', '<robot name="proto6">']
    parts += [f'  <material name="{n}"><color rgba="{c}"/></material>' for n, c in COLORS.items()]

    parts.append(link("base", "cyl", (b["r"], b["h"]), b["mass"], (0, 0, b["h"] / 2), "dark"))
    parts.append(joint("j1", "base", "shoulder", (0, 0, b["h"]), "0 0 1"))
    parts.append(link("shoulder", "cyl", (s["r"], s["h"]), s["mass"], (0, 0, s["h"] / 2), "orange"))
    parts.append(joint("j2", "shoulder", "upper", (0, 0, s["h"]), "0 1 0"))
    parts.append(link("upper", "box", (u["w"], u["w"] * 0.8, u["l"]), u["mass"], (0, 0, u["l"] / 2), "orange"))
    parts.append(joint("j3", "upper", "fore_a", (0, 0, u["l"]), "0 1 0"))
    parts.append(link("fore_a", "cyl", (f["r"], J4_OFFSET), f["mass"] * 0.6, (0, 0, J4_OFFSET / 2), "grey"))
    parts.append(joint("j4", "fore_a", "fore_b", (0, 0, J4_OFFSET), "0 0 1"))
    rest = f["l"] - J4_OFFSET
    parts.append(link("fore_b", "cyl", (f["r"] * 0.9, rest), f["mass"] * 0.4, (0, 0, rest / 2), "orange"))
    parts.append(joint("j5", "fore_b", "wrist", (0, 0, rest), "0 1 0"))
    parts.append(link("wrist", "cyl", (w["r"], w["l"]), w["mass"], (0, 0, w["l"] / 2), "grey"))
    parts.append(joint("j6", "wrist", "flange", (0, 0, w["l"]), "0 0 1"))
    parts.append(link("flange", "cyl", (fl["r"], fl["l"]), fl["mass"], (0, 0, fl["l"] / 2), "dark"))
    parts.append(joint("palm_fix", "flange", "palm", (0, 0, fl["l"]), "0 0 1", "fixed"))
    parts.append(link("palm", "box", palm, g["mass"], (0, 0, palm[2] / 2), "dark"))

    # Paralel parmaklar: y ekseninde kayar, 0 = kapalı, stroke = açık
    for side, sign in (("left", 1), ("right", -1)):
        parts.append(f"""
  <joint name="finger_{side}" type="prismatic">
    <parent link="palm"/><child link="finger_{side}"/>
    <origin xyz="0 {sign * fin[1] / 2:.4f} {palm[2]}" rpy="0 0 0"/>
    <axis xyz="0 {sign} 0"/>
    <limit lower="0" upper="{g["stroke"]}" effort="40" velocity="0.2"/>
    <dynamics damping="2.0" friction="0.1"/>
  </joint>""")
        parts.append(link(f"finger_{side}", "box", fin, g["finger_mass"], (0, 0, fin[2] / 2), "grey"))

    # Kavrama noktası: parmak uçlarının ortası (IK bu noktayı hedefler)
    tip = fl["l"] + palm[2] + fin[2] - 0.012
    parts.append(joint("grasp_fix", "flange", "grasp_target", (0, 0, tip), "0 0 1", "fixed"))
    parts.append('  <link name="grasp_target"><inertial><mass value="0.001"/>'
                 '<inertia ixx="1e-7" iyy="1e-7" izz="1e-7" ixy="0" ixz="0" iyz="0"/></inertial></link>')
    parts.append("</robot>\n")
    return "\n".join(parts)


def reach():
    P = PARAMS
    return P["upper"]["l"] + P["fore"]["l"] + P["wrist"]["l"] + P["flange"]["l"]


if __name__ == "__main__":
    out = Path(__file__).with_name("proto6.urdf")
    out.write_text(build(), encoding="utf-8")
    total = sum(v["mass"] for v in PARAMS.values()) + 2 * PARAMS["gripper"]["finger_mass"]
    print(f"{out}  |  omuzdan flanşa erişim {reach() * 1000:.0f} mm  |  toplam kütle {total:.2f} kg")
