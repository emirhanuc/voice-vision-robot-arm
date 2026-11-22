"""Mikrofon/Whisper gerektirmeyen ses yardımcıları testleri."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voicearm.stt.whisper_stt import BLOCK, SAMPLE_RATE, EnergyVAD, load_wav, save_wav, wer  # noqa: E402


def test_wer():
    assert wer("kirmizi kupu al", "kirmizi kupu al") == 0
    assert np.isclose(wer("kirmizi kupu al", "kirmizi kup al"), 1 / 3)


def test_wav_roundtrip(tmp_path):
    x = (0.5 * np.sin(np.linspace(0, 100, SAMPLE_RATE))).astype(np.float32)
    save_wav(tmp_path / "a.wav", x)
    assert np.allclose(load_wav(tmp_path / "a.wav"), x, atol=1e-4)


def test_vad_cuts_speech_and_stops_on_silence():
    rng = np.random.default_rng(0)
    noise = lambda n: (0.002 * rng.standard_normal(n * BLOCK)).astype(np.float32)  # noqa: E731
    tone = (0.3 * np.sin(np.arange(40 * BLOCK) / 5)).astype(np.float32)           # 1.2 sn "konuşma"
    signal = np.concatenate([noise(30), tone, noise(200)])
    blocks = (signal[i:i + BLOCK] for i in range(0, len(signal), BLOCK))
    seg = EnergyVAD().segment(blocks, noise_level=0.002)
    dur = len(seg) / SAMPLE_RATE
    assert 1.2 <= dur <= 1.2 + 0.3 + 0.8 + 0.05      # konuşma + ön-kayıt + sessizlik payı
