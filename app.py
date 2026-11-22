"""Gradio arayüzü: mikrofon veya metinle komut, robotun kamerası ve hareketin videosu.

    python app.py                       # http://127.0.0.1:7860
    python app.py --llm --whisper medium
    python app.py --robot proto6        # kendi 6 eksenli kolum
"""
import argparse
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from voicearm.perception import ColorDetector  # noqa: E402
from voicearm.sim.recorder import Recorder  # noqa: E402
from voicearm.world import build_world  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--llm", action="store_true")
ap.add_argument("--model", default="qwen2.5:7b")
ap.add_argument("--whisper", default="small")
ap.add_argument("--detector", choices=["color", "owl"], default="color")
ap.add_argument("--robot", choices=["panda", "proto6"], default="panda")
ap.add_argument("--share", action="store_true")
ARGS = ap.parse_args()

LOGS = Path("logs")
LOGS.mkdir(exist_ok=True)


class World:
    def __init__(self):
        self.stt = None
        self.detector = None
        self.reset(None)

    def reset(self, seed):
        import pybullet as p
        if p.isConnected():
            p.disconnect()
        if self.detector is None:
            if ARGS.detector == "owl":
                from voicearm.perception import OwlDetector
                self.detector = OwlDetector()
            else:
                self.detector = ColorDetector()
        w = build_world(robot=ARGS.robot, seed=seed, detector=self.detector, llm=ARGS.llm, model=ARGS.model,
                        log=False, camera_png=LOGS / "camera.png")
        self.scene, self.robot, self.perception, self.agent = w.scene, w.robot, w.perception, w.agent
        cam = self.scene.spec.layout
        self.rec = Recorder(every=10, eye=(cam.camera_eye[0] + 0.2, 0.25, cam.camera_eye[2]),
                            target=(cam.camera_target[0] - 0.05, 0.0, 0.72))
        self.robot.on_step = self.rec
        self.perception.observe()
        self.rec.snap()

    def transcribe(self, audio):
        if self.stt is None:
            from voicearm.stt.whisper_stt import WhisperSTT
            self.stt = WhisperSTT(ARGS.whisper)
        sr, data = audio
        data = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.abs(data).max() > 1.0:
            data /= 32768.0
        if sr != 16000:   # basit yeniden örnekleme
            t_new = np.linspace(0, len(data) / sr, int(len(data) * 16000 / sr), endpoint=False)
            data = np.interp(t_new, np.arange(len(data)) / sr, data).astype(np.float32)
        return self.stt.transcribe(data)


W = World()


def run_command(text, history):
    history = history or []
    if not text or not text.strip():
        return history, gr.update(), gr.update(), ""
    history.append({"role": "user", "content": text})
    W.rec.clear()
    W.rec.caption = f'Kullanıcı: "{text}"'
    t0 = time.time()
    replies = W.agent.handle(text)
    for r in replies:
        history.append({"role": "assistant", "content": r})
    W.rec.caption = "Robot: " + replies[-1].lstrip("✓?✗ ").split(" (")[0]
    for _ in range(10):
        W.rec.snap()
    gif = W.rec.save_gif(LOGS / f"action_{int(t0)}.gif", fps=12, speedup=2)
    return history, str(LOGS / "camera.png"), gif, ""


def run_voice(audio, history):
    if audio is None:
        return history, gr.update(), gr.update(), ""
    try:
        text = W.transcribe(audio)
    except ImportError:
        history = (history or []) + [{"role": "assistant",
                                      "content": "Ses için: python -m pip install faster-whisper"}]
        return history, gr.update(), gr.update(), ""
    if not text:
        return (history or []) + [{"role": "assistant", "content": "Sesi anlayamadım."}], \
            gr.update(), gr.update(), ""
    return run_command(text, history)


def reset(seed_text):
    seed = int(seed_text) if str(seed_text).strip().isdigit() else None
    W.reset(seed)
    return [], str(LOGS / "camera.png"), None


with gr.Blocks(title="Sesli Robot Kol") as demo:
    gr.Markdown("## 🎙️🦾 Sesli Robot Kol\nTürkçe komut ver: *\"kırmızı küpü mavi kutuya koy\"*, "
                "*\"önce yeşili al, sonra onu kırmızı kutuya bırak\"*. Robot emin değilse soru sorar.")
    with gr.Row():
        with gr.Column(scale=5):
            chat = gr.Chatbot(height=380, label="Diyalog")
            with gr.Row():
                txt = gr.Textbox(placeholder="Komut yaz ve Enter'a bas…", scale=4, show_label=False)
                send = gr.Button("Gönder", variant="primary", scale=1)
            mic = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Sesli komut (kaydet → durdur)")
            with gr.Row():
                seed = gr.Textbox(label="Sahne tohumu (boş = varsayılan)", scale=3)
                rst = gr.Button("Sahneyi sıfırla", scale=1)
        with gr.Column(scale=6):
            video = gr.Image(label="Son hareket", type="filepath", height=390)
            cam = gr.Image(value=str(LOGS / "camera.png"), label="Robotun kamerası + tespitler",
                           type="filepath", height=300)

    outs = [chat, cam, video, txt]
    txt.submit(run_command, [txt, chat], outs)
    send.click(run_command, [txt, chat], outs)
    mic.stop_recording(run_voice, [mic, chat], outs)
    rst.click(reset, [seed], [chat, cam, video])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(share=ARGS.share)
