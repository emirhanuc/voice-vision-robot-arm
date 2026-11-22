"""Doğal dil komutu -> yapılandırılmış görev listesi.

    "önce yeşil küpü al, sonra onu mavi kutuya koy"
      -> [Task(pick, yesil kup), Task(place, hedef=mavi kutu)]

İki arka uç, aynı çıktı:
  - RuleParser  : bağımlılıksız, Türkçe kalıplara dayalı (temel çizgi)
  - OllamaParser: lokal LLM (örn. qwen2.5:7b), JSON şemasıyla
`scripts/eval_parser.py` ikisini aynı veri setinde kıyaslar.
"""
import json
import re
import unicodedata
from dataclasses import asdict, dataclass

# Algının tanıdığı renkler + tanımadığı ama "renk" olduğunu bildiğimiz kelimeler.
# Tanınmayan renk olduğu gibi geçer; algı "görmedim" der (OWLv2 bunları da bulabilir).
COLOR_WORDS = {
    "kirmizi": ["kirmizi", "kizil", "red"],
    "yesil": ["yesil", "green"],
    "mavi": ["mavi", "blue"],
    "sari": ["sari", "yellow"],
    "turuncu": ["turuncu", "orange"],
    "mor": ["mor"],
    "pembe": ["pembe", "pink"],
    "siyah": ["siyah", "black"],
    "beyaz": ["beyaz", "white"],
}
KIND_WORDS = {
    "kup": ["kup", "blok", "kutucuk", "cube", "block"],
    "kutu": ["kutu", "kase", "tepsi", "sepet", "box", "tray"],
}
PICK_VERBS = ("al", "tut", "kaldir", "kavra", "yakala", "pick")
PLACE_VERBS = ("koy", "birak", "yerlestir", "tasi", "put", "place")
HOME_WORDS = ("basa", "baslangic", "eve", "home", "geri cekil", "dinlen")
REF_OBJ = ("onu", "bunu", "it")             # "onu mavi kutuya koy"
REF_TARGET = ("onun", "bunun", "ona")         # "kırmızıyı onun üstüne koy"
STEP_SPLIT = r"\b(?:ve sonra|daha sonra|sonra|ardindan|then)\b|[;.]"


@dataclass
class Task:
    action: str                       # "pick" | "place" | "pick_place" | "home"
    object_color: str | None = None
    object_kind: str | None = None
    target_color: str | None = None
    target_kind: str | None = None

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)


def normalize(text: str) -> str:
    text = text.lower().replace("ı", "i").replace("İ", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def _match(word: str, table: dict):
    for key, variants in table.items():
        if any(word.startswith(v) for v in variants):
            return key
    return None


def _is_verb(word: str, verbs) -> bool:
    # "al", "alip", "aliver" ... ama "altina" (altına) fiil değil
    return any(word == v or (word.startswith(v) and not word.startswith(("alt", "ard"))) for v in verbs)


def extract_color(text: str) -> str | None:
    for w in re.findall(r"\w+", normalize(text)):
        c = _match(w, COLOR_WORDS)
        if c:
            return c
    return None


class RuleParser:
    def parse(self, text: str) -> list[Task]:
        norm = normalize(text)
        if any(h in norm for h in HOME_WORDS) and not any(_is_verb(w, PLACE_VERBS) for w in norm.split()):
            return [Task(action="home")]
        steps = self._split(norm)
        tasks, last_obj = [], None
        for s in steps:
            t = self._parse_step(s, last_obj)
            if t is None:
                continue
            tasks.append(t)
            if t.object_color or t.object_kind:
                last_obj = (t.object_color, t.object_kind)
        return tasks

    @staticmethod
    def _split(norm: str) -> list[str]:
        """Adımlara böl. Fiilsiz virgül parçası, sonraki parçanın fiilini ödünç alır:
        "kırmızıyı mavi kutuya, yeşili kırmızı kutuya koy" -> iki adım."""
        parts = []
        for chunk in re.split(STEP_SPLIT, norm):
            parts += [p.strip() for p in chunk.split(",") if p.strip()]

        def verb(p):
            for w in p.split():
                if _is_verb(w, PLACE_VERBS) or _is_verb(w, PICK_VERBS):
                    return w
            return None

        for i in range(len(parts) - 2, -1, -1):
            if verb(parts[i]) is None and verb(parts[i + 1]) and any(
                    _match(w, COLOR_WORDS) or _match(w, KIND_WORDS) for w in parts[i].split()):
                parts[i] += " " + verb(parts[i + 1])
        merged = []   # içinde renk/tür/fiil olmayan parçaları (ör. "tamam") at
        for p in parts:
            ws = p.split()
            if any(_match(w, COLOR_WORDS) or _match(w, KIND_WORDS) or verb(p) or w in REF_OBJ + REF_TARGET
                   for w in ws) or any(h in p for h in HOME_WORDS):
                merged.append(p)
        return merged

    def _parse_step(self, s: str, last_obj):
        words = [w for w in re.findall(r"\w+", s) if w != "once"]
        if not words:
            return None
        if any(h in s for h in HOME_WORDS):
            return Task(action="home")

        # Nesne/hedef "ifadeleri": [renk] [tür] gruplarını sırayla çıkar
        phrases, i = [], 0
        while i < len(words):
            w = words[i]
            color, kind = _match(w, COLOR_WORDS), _match(w, KIND_WORDS)
            if w in REF_TARGET:
                phrases.append({"ref": "target", "i": i})
            elif w in REF_OBJ:
                phrases.append({"ref": "obj", "i": i})
            elif color:
                nxt = _match(words[i + 1], KIND_WORDS) if i + 1 < len(words) else None
                phrases.append({"color": color, "kind": nxt, "i": i + (1 if nxt else 0),
                                "word": words[i + 1] if nxt else w})
                i += 1 if nxt else 0
            elif kind:
                phrases.append({"color": None, "kind": kind, "i": i, "word": w})
            i += 1

        has_place = any(_is_verb(w, PLACE_VERBS) for w in words)
        has_pick = any(_is_verb(w, PICK_VERBS) for w in words)

        # Hedef: yönelme/bulunma eki alan ifade (kutuYA, kutunun içiNE, küpün üstüNE) ya da "onun/ona"
        target, obj = None, None
        for ph in phrases:
            if ph.get("ref") == "target" or self._is_target_word(ph.get("word", ""), words, ph["i"]):
                target = ph
            elif obj is None:
                obj = ph
        if obj is None and target is not None and not has_place:
            obj, target = target, None   # "kutuyu al" gibi

        def resolve(ph, default_kind):
            if ph is None:
                return None, None
            if "ref" in ph:
                return last_obj if last_obj else (None, None)
            return ph["color"], ph["kind"] or default_kind

        oc, ok = resolve(obj, "kup")
        tc, tk = resolve(target, "kutu")
        if target is not None and "ref" not in target and target["kind"] is None:
            after = words[target["i"] + 1] if target["i"] + 1 < len(words) else ""
            tk = "kup" if after.startswith(("ust", "uzer")) else "kutu"

        if has_place or target is not None:
            if obj is None or (obj.get("ref") == "obj" and not has_pick):
                # "mavi kutuya koy" / "onu mavi kutuya koy" -> elimdekini bırak
                if obj is None or "ref" in obj:
                    return Task("place", target_color=tc, target_kind=tk)
            return Task("pick_place", oc, ok, tc, tk)
        return Task("pick", oc, ok or "kup")

    @staticmethod
    def _is_target_word(word, words, idx):
        # "kutuya", "kutusuna", "kupun" + "ustune/icine/uzerine", "kutuda"
        if re.search(r"(y?a|y?e|na|ne)$", word) and _match(word, KIND_WORDS):
            return True
        nxt = words[idx + 1] if idx + 1 < len(words) else ""
        return nxt.startswith(("ust", "uzer", "ic"))


class OllamaParser:
    SYSTEM = (
        "Bir robot kol için Türkçe komut ayrıştırıcısın. Komutu adımlara böl ve SADECE şu JSON'u döndür:\n"
        '{"steps": [{"action": "pick"|"place"|"pick_place"|"home", "object_color": str|null, '
        '"object_kind": "kup"|"kutu"|null, "target_color": str|null, "target_kind": "kup"|"kutu"|null}]}\n'
        "Kurallar:\n"
        "- pick: nesneyi al ve tut. place: elde tutulanı hedefe bırak. pick_place: al ve hedefe koy.\n"
        "- Renkler Türkçe karaktersiz küçük harf: kirmizi, yesil, mavi, sari, turuncu, mor, pembe, siyah, beyaz.\n"
        "- küp/blok -> kup ; kutu/tepsi/sepet -> kutu. Tür söylenmemişse nesne için kup, hedef için kutu.\n"
        "- 'onu', 'onun üstüne' gibi zamirleri önceki adımdaki nesnenin rengi/türüyle doldur.\n"
        "- Belirtilmeyen alanı null bırak, tahmin etme. Başka metin yazma."
    )

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        self.model, self.host = model, host

    def parse(self, text: str) -> list[Task]:
        import requests  # isteğe bağlı bağımlılık
        r = requests.post(f"{self.host}/api/chat", json={
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": self.SYSTEM},
                         {"role": "user", "content": text}],
        }, timeout=120)
        r.raise_for_status()
        return self.from_json(r.json()["message"]["content"])

    @staticmethod
    def from_json(content: str) -> list[Task]:
        data = json.loads(content)
        steps = data.get("steps", [data]) if isinstance(data, dict) else data
        fields = Task.__dataclass_fields__
        out = []
        for st in steps:
            st = {k: (normalize(v) if isinstance(v, str) else v) for k, v in st.items() if k in fields}
            if st.get("action") in {"pick", "place", "pick_place", "home"}:
                out.append(Task(**st))
        return out
