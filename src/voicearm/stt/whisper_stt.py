"""Mikrofon -> metin (faster-whisper) + enerji tabanlı konuşma algılama (VAD).

Kayıt, konuşma başlayınca başlar ve ~0.8 sn sessizlikte kendiliğinden biter;
sabit süre beklemek yok. Kurulum: pip install faster-whisper sounddevice
"""
import queue
import wave

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 480   # 30 ms

# Whisper'ı alan kelimelerine yönlendirir (Türkçe renk/fiil tanıma doğruluğunu artırır)
DOMAIN_PROMPT = ("Robot kol komutları: kırmızı küpü mavi kutuya koy. yeşil küpü al. sarı küpü bırak. "
                 "küpün üstüne koy. önce, sonra, eve dön.")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


class EnergyVAD:
    """Ortam gürültüsünü ölçer; eşik = gürültü × oran. Konuşma bitince kaydı keser."""

    def __init__(self, ratio=3.0, min_level=0.008, silence_s=0.8, max_s=10.0, preroll_s=0.3):
        self.ratio, self.min_level = ratio, min_level
        self.silence_blocks = int(silence_s * SAMPLE_RATE / BLOCK)
        self.max_blocks = int(max_s * SAMPLE_RATE / BLOCK)
        self.preroll = int(preroll_s * SAMPLE_RATE / BLOCK)

    def segment(self, blocks, noise_level: float) -> np.ndarray | None:
        """blocks: 30 ms'lik bloklar akışı. Konuşma bölümünü döndürür."""
        thr = max(noise_level * self.ratio, self.min_level)
        history, speech, silent, started = [], [], 0, False
        for i, b in enumerate(blocks):
            loud = rms(b) > thr
            if not started:
                history = (history + [b])[-self.preroll:]
                if loud:
                    started, speech = True, list(history)
                elif i > self.max_blocks:
                    return None
                continue
            speech.append(b)
            silent = 0 if loud else silent + 1
            if silent >= self.silence_blocks or len(speech) >= self.max_blocks:
                break
        return np.concatenate(speech) if speech else None


class WhisperSTT:
    def __init__(self, model_size: str = "small", device: str = "auto", vad: EnergyVAD | None = None):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.vad = vad or EnergyVAD()

    def record(self) -> np.ndarray:
        import sounddevice as sd
        q: queue.Queue = queue.Queue()
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
                            callback=lambda data, *_: q.put(data[:, 0].copy())):
            noise = np.median([rms(q.get()) for _ in range(15)])   # ilk ~0.45 sn: ortam gürültüsü
            print("🎤 Dinliyorum... (konuş, susunca kendisi durur)")

            def stream():
                while True:
                    yield q.get()
            audio = self.vad.segment(stream(), noise)
        return audio if audio is not None else np.zeros(SAMPLE_RATE, np.float32)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language="tr", beam_size=5, vad_filter=True,
                                            initial_prompt=DOMAIN_PROMPT, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def listen(self) -> str:
        return self.transcribe(self.record())


def save_wav(path, audio: np.ndarray):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def load_wav(path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32767


def wer(ref: str, hyp: str) -> float:
    """Kelime hata oranı (Levenshtein / referans kelime sayısı)."""
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0], d[0, :] = range(len(r) + 1), range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(len(r), 1)
