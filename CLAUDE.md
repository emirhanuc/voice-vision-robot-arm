# Voice- and Vision-Guided Robot Arm (Simulation) — github.com/emirhanuc/voice-vision-robot-arm

Türkçe sesli komutla, kamerayla gören ve computed-torque ile kontrol edilen robot kol. CV/GitHub portföy projesi.
Geliştirici: Emirhan (Yapay Zeka Mühendisliği 3. sınıf). Türkçe, kısa ve net iletişim tercih eder.

## Mimari
Ses → `stt/whisper_stt.py` (EnergyVAD + faster-whisper, DOMAIN_PROMPT) → metin → `nlp/parser.py`
(RuleParser | OllamaParser → `list[Task]`) → `executor.Agent` → `perception/` + `sim/robot.py`

- **Task**: action ∈ pick | place | pick_place | home; object_color/kind, target_color/kind. kind ∈ kup | kutu.
  Renkler ASCII: kirmizi, yesil, mavi, sari (+ turuncu, mor… ayrıştırılır ama HSV dedektörü tanımaz).
- **Agent** (`executor.py`): `handle(text) -> list[str]`; cevaplar "✓ / ? / ✗" ile başlar. Belirsizlikte `Clarify`
  fırlatır, `pending` dolar, bir sonraki mesaj cevap sayılır (renk, sıra no, "iptal"). `held` elde tutulanı izler.
  pick_place'te hedef, nesne alınmadan ÖNCE çözülür. İstifleme: hedef kind=kup.
- **Algı** (`perception/`): `Camera.capture() -> Frame(rgb, points)` (robot tabanı koordinatları). `ColorDetector` (HSV) |
  `OwlDetector` (OWLv2). `Detection.position` = üst yüzey merkezi; küp/kutu ayrımı metrik genişlik (<9 cm küp).
  `Perception.observe()` kolu OBSERVE_POSE'a çeker, yüksekliği >20 cm olanları (tutulan küp) atar.
- **Kollar** (`sim/robots.py`): `RobotSpec` + `Layout` (çalışma alanı, kamera, gözlem/ev pozu). `panda` (7 DOF,
  control="torque") ve `proto6` (Emirhan'ın 6 DOF kolu, control="stepper", URDF: `robots/proto6/generate_urdf.py`,
  parametreler TAHMİNİ). `world.build_world(robot=...)` her şeyi kurar; script'lerde `--robot`.
- **Kontrol** (`sim/robot.py` ArmSim): Kartezyen çizgi + quintic → IK (spec.down_euler yönelimi) ile q_d, J⁺ ile q̇_d.
  torque: computed-torque (PyBullet inverse dynamics), dahili motorlar kapalı. stepper: adım çözünürlüğüne yuvarlanmış
  POSITION_CONTROL, eklem başına tork/hız limiti ve pos_gain; `saturation_time` = tork limitinde geçen süre
  (adım kaçırma riski). 240 Hz. Parmaklar JOINT_GEAR ile bağlı (nesne ortalanır). Kavrama = iki parmakta temas.
- `RobotInterface` soyut; gerçek kol (STM32 seri port) aynı arayüzü uygulayacak.

## Kurallar
- Executor/Agent nesne konumlarını SADECE algıdan alır. Sim ground-truth yalnızca `scripts/evaluate.py` ve testlerde.
- Yeni özellik → `data/commands.jsonl`'e örnek + test. `python -m pytest -q tests` yeşil kalmalı.
- README'deki metrikler script çıktısıyla güncellenir, elle uydurulmaz.

## Komutlar
- `python app.py` — Gradio arayüzü (gradio>=6)
- `python scripts/run.py [--robot proto6] [--voice] [--llm] [--detector owl] [--seed N] [--headless -c "..."]`
- `python -m pytest -q tests` · `python scripts/evaluate.py --trials 20` · `python scripts/eval_parser.py [--parser llm]`
- `python scripts/record_voice_set.py` → `python scripts/eval_voice.py` · `python scripts/plot_tracking.py` · `python scripts/make_demo_gif.py`
- Windows: `pip` yerine `python -m pip`; Python 3.11 venv.

## Durum
- [x] Sahne, computed-torque kontrol, temas tabanlı kavrama
- [x] RGB-D algı (HSV + OWLv2), pikselden 3D, kamerayla doğrulama — 20/20, 3.6 mm
- [x] Çok adımlı komut, zamir, istifleme, geri sorma diyaloğu — kural ayrıştırıcı %96 (50 komut)
- [x] VAD'li mikrofon, sesli test seti araçları, Gradio arayüzü, demo GIF, CI
- [x] İstifleme: parmak bağı + tepsi merkezi düzeltmesi (konum hatası 3.6 → 2.0 mm), 4 katlı kule testi
- [x] proto6: parametrik URDF, step motor modeli, iki kolda 20/20; İngilizce README + README.tr.md
- [ ] Kullanıcı makinesinde ölçülecek: LLM ayrıştırıcı, Whisper WER — sonra `scripts/update_readme_metrics.py`
- [ ] proto6: gerçek ölçülen kütle/tork değerleri, CadQuery STL'leri (`MESHES`), STM32 seri `RobotInterface`
- [ ] RealSense + el-göz kalibrasyonu ile gerçek kamera
