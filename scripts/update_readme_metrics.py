"""Kendi makinende ölçtüğün LLM ve ses sonuçlarını README.md / README.tr.md tablolarına yazar.

    python scripts/eval_parser.py --parser llm
    python scripts/eval_voice.py
    python scripts/update_readme_metrics.py
Hangi sonuç dosyası varsa onu kullanır (docs/eval_parser_llm_*.json, docs/eval_voice_*.json).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def latest(pattern):
    files = sorted(DOCS.glob(pattern), key=lambda f: f.stat().st_mtime)
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def set_row(text, label_regex, value):
    """Tablo satırında label'dan sonraki ilk hücreyi değiştirir."""
    return re.sub(rf"(\| {label_regex}[^|]*\| )[^|]*(\|)", rf"\g<1>{value} \g<2>", text)


def main():
    llm = latest("eval_parser_llm_*.json")
    voice = latest("eval_voice_*.json")
    if not (llm or voice):
        print("Sonuç dosyası yok. Önce eval_parser.py --parser llm ve/veya eval_voice.py çalıştır.")
        return
    for name, labels in {"README.md": ("Command parsing, local LLM", "Speech → command accuracy"),
                         "README.tr.md": ("Komut ayrıştırma, yerel LLM", "Ses → komut doğruluğu")}.items():
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        tr = name == "README.tr.md"

        def pct(x, digits=0):
            v = f"{100 * x:.{digits}f}"
            return f"%{v}" if tr else f"{v} %"
        if llm:
            text = set_row(text, re.escape(labels[0]),
                           f"**{pct(llm['accuracy'])}** ({llm['parser'].split(':', 1)[1]}, "
                           f"{llm['mean_latency_ms'] / 1000:.1f} s/{'komut' if tr else 'cmd'})")
        if voice:
            text = set_row(text, re.escape(labels[1]),
                           f"**{pct(voice['command_accuracy'])}**, WER {pct(voice['wer'], 1)} "
                           f"(whisper-{voice['whisper']})")
        path.write_text(text, encoding="utf-8")
        print(f"güncellendi: {name}")


if __name__ == "__main__":
    main()
