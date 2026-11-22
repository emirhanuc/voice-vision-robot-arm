"""Komut satırından robotu yönet (metin veya ses).

    python scripts/run.py                         # GUI + metin
    python scripts/run.py --voice                 # mikrofon (faster-whisper)
    python scripts/run.py --llm                   # ayrıştırıcı: Ollama (qwen2.5:7b)
    python scripts/run.py --detector owl          # OWLv2 açık sözlüklü tespit
    python scripts/run.py --seed 7                # rastgele nesne yerleşimi
    python scripts/run.py --robot proto6          # kendi 6 eksenli kolum (step motorlu)
    python scripts/run.py --headless -c "kırmızı küpü mavi kutuya koy" -c "eve dön"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voicearm.world import build_world  # noqa: E402


def build_agent(headless=False, llm=False, model="qwen2.5:7b", detector="color", seed=None,
                robot="panda", camera_png="logs/camera.png"):
    w = build_world(robot=robot, gui=not headless, seed=seed, detector=detector, llm=llm, model=model,
                    camera_png=camera_png)
    return w.scene, w.robot, w.agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--llm", action="store_true", help="Ollama ile ayrıştır")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--voice", action="store_true", help="Mikrofondan komut al")
    ap.add_argument("--whisper", default="small", help="tiny | base | small | medium")
    ap.add_argument("--detector", choices=["color", "owl"], default="color")
    ap.add_argument("--seed", type=int, default=None, help="Rastgele sahne")
    ap.add_argument("--robot", choices=["panda", "proto6"], default="panda")
    ap.add_argument("-c", "--command", action="append", help="Komut (birden fazla verilebilir)")
    ap.add_argument("--log", default="logs/last_run.json", help="Kontrol kaydı (grafik için)")
    args = ap.parse_args()

    _, robot, agent = build_agent(args.headless, args.llm, args.model, args.detector, args.seed, args.robot)
    listen = None
    if args.voice:
        from voicearm.stt.whisper_stt import WhisperSTT
        listen = WhisperSTT(args.whisper).listen

    def handle(text):
        for line in agent.handle(text):
            print(f"  🤖 {line}")
        if agent.last_tasks and not agent.pending:
            print(f"     görevler: {[t.to_dict() for t in agent.last_tasks]}")

    def save_log():
        path = Path(args.log)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(robot.log))
        print(f"Kontrol kaydı: {path}  (grafik: python scripts/plot_tracking.py)")
        if robot.spec.control == "stepper":
            sat = ", ".join(f"J{i + 1}:{t:.2f}s" for i, t in enumerate(robot.saturation_time))
            print(f"Step motor tork limitinde geçen süre (adım kaçırma riski): {sat}")

    try:
        if args.command:
            for c in args.command:
                print(f"> {c}")
                handle(c)
            return
        print("Komut ver (çıkmak için 'q'). Örnek: kırmızı küpü mavi kutuya koy")
        print("Robot soru sorarsa cevabını yaz/söyle; 'iptal' ile vazgeçebilirsin.")
        while True:
            if listen:
                text = listen()
                print(f"> {text}")
            else:
                text = input("> ")
            if text.strip().lower() in {"q", "çık", "cik", "exit"}:
                break
            if text.strip():
                handle(text)
    finally:
        save_log()


if __name__ == "__main__":
    main()
