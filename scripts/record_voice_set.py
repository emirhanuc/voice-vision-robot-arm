"""Sesli test setini kaydet: data/commands.jsonl'deki her komutu sen okursun, wav olarak saklanır.

    python scripts/record_voice_set.py                 # hepsi (~10 dk)
    python scripts/record_voice_set.py --start 20      # kaldığın yerden
Her komut için: ekrandaki cümleyi oku, susunca kayıt kendisi biter. 'r' + Enter = tekrar kaydet.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.stt.whisper_stt import BLOCK, SAMPLE_RATE, EnergyVAD, rms, save_wav  # noqa: E402


def record_once(vad):
    import queue

    import sounddevice as sd
    q = queue.Queue()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
                        callback=lambda d, *_: q.put(d[:, 0].copy())):
        noise = np.median([rms(q.get()) for _ in range(15)])
        print("   🎤 konuş...")

        def stream():
            while True:
                yield q.get()
        return vad.segment(stream(), noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/commands.jsonl")
    ap.add_argument("--out", default="data/voice")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vad = EnergyVAD()
    for i, r in enumerate(rows):
        if i < args.start:
            continue
        while True:
            input(f"\n[{i + 1}/{len(rows)}] Enter'a bas ve oku:  «{r['text']}»")
            audio = record_once(vad)
            if audio is None:
                print("   ses algılanmadı, tekrar.")
                continue
            print(f"   {len(audio) / SAMPLE_RATE:.1f} sn kaydedildi.", end=" ")
            if input("Tamam mı? (Enter = evet, r = tekrar) ").strip().lower() != "r":
                break
        save_wav(out / f"{i:03d}.wav", audio)
    print(f"\nBitti: {out}/  ->  python scripts/eval_voice.py")


if __name__ == "__main__":
    main()
