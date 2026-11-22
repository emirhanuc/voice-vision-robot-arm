"""Ses -> komut uçtan uca değerlendirme (record_voice_set.py ile kaydedilen wav'lar üzerinde).

    python scripts/eval_voice.py                              # whisper small + kural ayrıştırıcı
    python scripts/eval_voice.py --whisper medium --parser llm
Ölçütler: WER (kelime hata oranı), komut doğruluğu (ses -> JSON görevler birebir doğru), gecikme.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.nlp.parser import OllamaParser, RuleParser, normalize  # noqa: E402
from voicearm.stt.whisper_stt import WhisperSTT, load_wav, wer  # noqa: E402


def clean(t):
    return " ".join("".join(c if c.isalnum() or c == " " else " " for c in normalize(t)).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/commands.jsonl")
    ap.add_argument("--voice", default="data/voice")
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--parser", choices=["rule", "llm"], default="rule")
    ap.add_argument("--model", default="qwen2.5:7b")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    stt = WhisperSTT(args.whisper)
    parser = RuleParser() if args.parser == "rule" else OllamaParser(model=args.model)

    wers, hits, stt_ms, n, errors = [], 0, [], 0, []
    for i, r in enumerate(rows):
        wav = Path(args.voice) / f"{i:03d}.wav"
        if not wav.exists():
            continue
        n += 1
        t0 = time.time()
        text = stt.transcribe(load_wav(wav))
        stt_ms.append(1000 * (time.time() - t0))
        wers.append(wer(clean(r["text"]), clean(text)))
        try:
            got = [t.to_dict() for t in parser.parse(text)]
        except Exception as e:
            got = str(e)
        ok = got == r["expected"]
        hits += ok
        print(f"{'✓' if ok else '✗'} «{r['text']}»  ->  «{text}»")
        if not ok:
            errors.append({"ref": r["text"], "hyp": text})

    if n == 0:
        sys.exit("Kayıt yok. Önce: python scripts/record_voice_set.py")
    res = {"whisper": args.whisper, "parser": args.parser, "n": n,
           "wer": float(np.mean(wers)), "command_accuracy": hits / n,
           "stt_latency_ms_mean": float(np.mean(stt_ms)), "errors": errors}
    print(f"\nWER %{100 * res['wer']:.1f} | komut doğruluğu {hits}/{n} = %{100 * hits / n:.1f} "
          f"| STT gecikmesi {res['stt_latency_ms_mean']:.0f} ms")
    Path("docs").mkdir(exist_ok=True)
    Path(f"docs/eval_voice_{args.whisper}_{args.parser}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
