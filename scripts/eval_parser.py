"""Komut ayrıştırıcı değerlendirmesi: kural tabanlı vs LLM, aynı veri setinde.

    python scripts/eval_parser.py                       # kural tabanlı
    python scripts/eval_parser.py --parser llm          # Ollama (qwen2.5:7b)
    python scripts/eval_parser.py --parser llm --model llama3.1:8b
Ölçüt: komutun TÜM adımları tüm alanlarıyla birebir doğruysa doğru sayılır (exact match).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.nlp.parser import OllamaParser, RuleParser  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", choices=["rule", "llm"], default="rule")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--data", default="data/commands.jsonl")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    parser = RuleParser() if args.parser == "rule" else OllamaParser(model=args.model)
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    per_cat = defaultdict(lambda: [0, 0])
    errors, latencies = [], []
    for r in rows:
        t0 = time.time()
        try:
            got = [t.to_dict() for t in parser.parse(r["text"])]
        except Exception as e:  # LLM bozuk JSON vb.
            got = f"HATA: {e}"
        latencies.append(time.time() - t0)
        ok = got == r["expected"]
        per_cat[r["category"]][0] += ok
        per_cat[r["category"]][1] += 1
        if not ok:
            errors.append((r["text"], r["expected"], got))

    total = sum(c[0] for c in per_cat.values())
    name = "rule" if args.parser == "rule" else f"llm:{args.model}"
    print(f"\n{name}: {total}/{len(rows)} = %{100 * total / len(rows):.1f}   "
          f"ort. gecikme {1000 * sum(latencies) / len(latencies):.0f} ms")
    for cat, (ok, n) in sorted(per_cat.items()):
        print(f"  {cat:18s} {ok:2d}/{n:<2d}")
    for text, exp, got in errors:
        print(f"\n  ✗ {text}")
        if args.verbose:
            print(f"    beklenen: {exp}\n    çıkan:    {got}")

    out = Path("docs") / f"eval_parser_{name.replace(':', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "parser": name, "accuracy": total / len(rows), "n": len(rows),
        "per_category": {k: v[0] / v[1] for k, v in per_cat.items()},
        "mean_latency_ms": 1000 * sum(latencies) / len(latencies),
        "errors": [e[0] for e in errors]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
