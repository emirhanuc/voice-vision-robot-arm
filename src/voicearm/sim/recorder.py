"""Simülasyondan video/GIF kaydı (demo ve Gradio arayüzü için)."""
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont


class Recorder:
    def __init__(self, width=520, height=390, every=12,
                 eye=(1.35, 0.25, 1.45), target=(0.38, 0.0, 0.72)):
        self.w, self.h, self.every = width, height, every
        self.view = p.computeViewMatrix(eye, target, [0, 0, 1])
        self.proj = p.computeProjectionMatrixFOV(50, width / height, 0.1, 5)
        self.frames: list[np.ndarray] = []
        self.caption = ""
        self._n = 0
        self.enabled = True

    def __call__(self):
        """PandaSim her simülasyon adımında çağırır."""
        self._n += 1
        if self.enabled and self._n % self.every == 0:
            self.snap()

    def snap(self):
        _, _, rgba, _, _ = p.getCameraImage(self.w, self.h, self.view, self.proj,
                                            renderer=p.ER_TINY_RENDERER)
        img = np.reshape(rgba, (self.h, self.w, 4))[:, :, :3].astype(np.uint8)
        self.frames.append(self._overlay(img) if self.caption else img)

    def _overlay(self, img):
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 17)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", 17)
            except OSError:
                font = ImageFont.load_default()
        lines, cur = [], ""
        for word in self.caption.split():
            trial = f"{cur} {word}".strip()
            if d.textlength(trial, font=font) > self.w - 20 and cur:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
        bar = 10 + 22 * len(lines)
        d.rectangle([0, self.h - bar, self.w, self.h], fill=(20, 20, 20))
        for i, line in enumerate(lines):
            d.text((10, self.h - bar + 5 + 22 * i), line, fill=(255, 255, 255), font=font)
        return np.asarray(im)

    def clear(self):
        self.frames = []

    def save_gif(self, path, fps=15, speedup=2, max_width=None):
        frames = self.frames[::speedup] if speedup > 1 else self.frames
        if not frames:
            return None
        ims = [Image.fromarray(f) for f in frames]
        if max_width and ims[0].width > max_width:
            h = int(ims[0].height * max_width / ims[0].width)
            ims = [im.resize((max_width, h), Image.LANCZOS) for im in ims]
        # Tüm karelere ORTAK palet: örnek karelerden bir mozaik oluşturup 256 renge indir.
        # (Kare başına adaptif palet renkleri bozuyor: sarı/yeşil küpler griye kayıyordu.)
        sample = ims[:: max(1, len(ims) // 12)]
        mosaic = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
        for i, im in enumerate(sample):
            mosaic.paste(im, (0, i * im.height))
        palette = mosaic.quantize(colors=256, method=Image.MEDIANCUT)
        ims = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in ims]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
        return path
