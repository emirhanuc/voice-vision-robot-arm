"""Görev yürütme + diyalog.

Agent, kullanıcıyla tur tur konuşur:
  - Komutu görev listesine çevirir, adımları sırayla yürütür.
  - Bilgi eksik/belirsizse SORU sorar ve bir sonraki mesajı cevap olarak kullanır
    ("Hangi küp? kırmızı, yeşil, mavi" -> "kırmızı olanı").
  - Elinde ne tuttuğunu hatırlar ("kırmızı küpü al" ... "mavi kutuya bırak").

Nesne konumları SADECE kameradan gelir (simülasyonun gerçek değerleri kullanılmaz);
aynı kod gerçek kol + RealSense ile çalışır. Başarı yeniden algılanarak doğrulanır.
"""
import re
from dataclasses import dataclass, field

import numpy as np

from .nlp.parser import Task, extract_color, normalize
from .perception import Detection, Perception
from .sim.robot import RobotInterface

NUMBERS = {"1": 0, "bir": 0, "birinci": 0, "ilk": 0, "2": 1, "iki": 1, "ikinci": 1,
           "3": 2, "uc": 2, "ucuncu": 2, "4": 3, "dort": 3, "dorduncu": 3}
CANCEL = ("iptal", "vazgec", "bosver", "cancel", "dur")
TR_NAME = {"kirmizi": "kırmızı", "yesil": "yeşil", "sari": "sarı", "kup": "küp"}


def pretty(name: str) -> str:
    return " ".join(TR_NAME.get(p, p) for p in name.split("_"))


class TaskError(Exception):
    pass


@dataclass
class Clarify(Exception):
    question: str
    slot: str                       # "object" | "target"
    options: list = field(default_factory=list)


def grasp_point(d: Detection):
    """Küpü yüksekliğinin ortasından kavra."""
    return [d.position[0], d.position[1], d.position[2] - d.height / 2]


class Agent:
    def __init__(self, parser, perception: Perception, robot: RobotInterface):
        self.parser, self.perception, self.robot = parser, perception, robot
        self.held: Detection | None = None
        self.queue: list[Task] = []
        self.pending: tuple[Task, Clarify] | None = None
        self.forced: dict = {}
        self.last_tasks: list[Task] = []

    # ---------------- diyalog ----------------
    def handle(self, text: str) -> list[str]:
        """Bir kullanıcı mesajını işler, robotun cevaplarını döndürür.
        Son cevap bir soruysa self.pending doludur ve sonraki mesaj cevap sayılır."""
        if self.pending:
            return self._answer(text)
        tasks = self.parser.parse(text)
        self.last_tasks = tasks
        if not tasks:
            return ["Komutu anlayamadım. Örnek: 'kırmızı küpü mavi kutuya koy'."]
        self.queue = list(tasks)
        return self._run()

    def _answer(self, text: str) -> list[str]:
        task, q = self.pending
        self.pending = None
        norm = normalize(text)
        if any(c in norm for c in CANCEL):
            self.queue.clear()
            return ["Tamam, iptal ettim."]
        idx = next((NUMBERS[w] for w in re.findall(r"\w+", norm) if w in NUMBERS), None)
        color = extract_color(text)
        if idx is not None and idx < len(q.options):
            self.forced[q.slot] = q.options[idx]
        elif color:
            if q.slot == "object":
                task.object_color = color
            else:
                task.target_color = color
                task.target_kind = task.target_kind or "kutu"
        else:
            self.pending = (task, q)
            return [f"Anlayamadım. {q.question}"]
        self.queue.insert(0, task)
        return self._run()

    def _run(self) -> list[str]:
        out = []
        while self.queue:
            task = self.queue.pop(0)
            try:
                out.append("✓ " + self.execute(task))
            except Clarify as q:
                self.pending = (task, q)
                out.append("? " + q.question)
                return out
            except TaskError as e:
                self.queue.clear()
                out.append("✗ " + str(e))
                return out
            finally:
                self.forced = {} if not self.pending else self.forced
        return out

    # ---------------- yürütme ----------------
    def _resolve(self, dets, color, kind, slot) -> Detection:
        if slot in self.forced:
            return self.forced[slot]
        exclude = self.held
        matches = [d for d in dets if d.kind == kind and (color is None or d.color == color)
                   and not (exclude is not None and d.color == exclude.color and d.kind == exclude.kind)]
        what = f"{pretty(color) + ' ' if color else ''}{pretty(kind)}"
        if not matches:
            seen = ", ".join(pretty(d.name) for d in dets) or "hiçbir şey"
            raise TaskError(f"Kamerada {what} görmedim. Gördüklerim: {seen}.")
        if len(matches) == 1:
            return matches[0]
        if color is None:
            opts = ", ".join(pretty(d.color) for d in matches)
            q = f"Hangi {pretty(kind)}?" if slot == "object" else "Hangisine bırakayım?"
            raise Clarify(f"{q} Gördüklerim: {opts}.", slot, matches)
        opts = "; ".join(f"{i + 1}) ({d.position[0]:.2f}, {d.position[1]:.2f})" for i, d in enumerate(matches))
        raise Clarify(f"Birden fazla {what} var, hangisi? {opts}", slot, matches)

    def execute(self, task: Task) -> str:
        if task.action == "home":
            self.robot.home()
            return "Başlangıç pozisyonuna döndüm."

        if task.action in ("pick", "pick_place") and self.held is not None:
            raise TaskError(f"Elimde zaten {pretty(self.held.name)} var. Önce nereye bırakacağımı söyle.")
        if task.action == "place" and self.held is None:
            raise TaskError("Elimde bir şey yok. Önce ne alacağımı söyle.")
        if task.action in ("place", "pick_place") and task.target_color is None and task.target_kind is None \
                and "target" not in self.forced:
            raise Clarify("Nereye bırakayım?", "target")

        dets = self.perception.observe()
        msgs = []
        if task.action in ("pick", "pick_place"):
            obj = self._resolve(dets, task.object_color, task.object_kind or "kup", "object")
            target = None
            if task.action == "pick_place":   # hedefi almadan ÖNCE çöz (soru sorulacaksa boşuna taşıma)
                target = self._resolve(dets, task.target_color, task.target_kind or "kutu", "target")
            if not self.robot.pick(grasp_point(obj)):
                self.robot.open_gripper()
                raise TaskError(f"{pretty(obj.name)} kavranamadı (parmaklarda temas kuvveti yok).")
            self.held = obj
            msgs.append(f"{pretty(obj.name)} tutuldu")
            if target is None:
                return msgs[0] + "."
            dets_target = target
        else:
            dets_target = self._resolve(dets, task.target_color, task.target_kind or "kutu", "target")

        return self._place(dets_target)

    def _place(self, target: Detection) -> str:
        held = self.held
        if target.kind == "kutu":
            table_z = target.position[2] - target.height
            self.robot.place([target.position[0], target.position[1], table_z])
        else:   # istifleme: hedef küpün üst yüzeyine, kendi yüksekliğinin yarısı + pay
            self.robot.place(list(target.position), drop_height=held.height / 2 + 0.008)
        self.held = None

        after = self.perception.observe()
        if target.kind == "kutu":
            in_tray = any(d.kind == "kup" and d.color == held.color and
                          np.linalg.norm(d.position[:2] - target.position[:2]) < 0.08 for d in after)
            filled = any(d.kind == "kutu" and np.linalg.norm(d.position[:2] - target.position[:2]) < 0.05
                         and d.height > target.height + 0.02 for d in after)
            ok = in_tray or filled
        else:
            ok = any(d.color == held.color and d.kind == "kup" and
                     np.linalg.norm(d.position[:2] - target.position[:2]) < 0.03 and
                     d.position[2] > target.position[2] + held.height / 2 for d in after)
        if not ok:
            raise TaskError(f"{pretty(held.name)} {pretty(target.name)} üzerinde/içinde görünmüyor.")
        verb = "içine" if target.kind == "kutu" else "üstüne"
        return f"{pretty(held.name)} → {pretty(target.name)} {verb} kondu (kamerayla doğrulandı)."
