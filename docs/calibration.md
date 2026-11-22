# Gerçek kameraya geçiş: el-göz kalibrasyonu

Simülasyonda kameranın robot tabanına göre konumu biliniyor. Gerçekte bunu ölçmen gerekir:
`T_base_cam` (4x4), kamera koordinatlarını robot tabanı koordinatlarına çevirir.

## Sabit kamera (eye-to-hand), ArUco ile
1. Gripper'a bir ArUco işaretçisi (ör. 5x5, 40 mm) sabitle.
2. Kolu kameranın gördüğü 15-20 farklı pozisyon/yönelime götür. Her birinde kaydet:
   - robotun ileri kinematiğinden `T_base_gripper`
   - `cv2.aruco` + `cv2.solvePnP` ile `T_cam_marker`
3. `cv2.calibrateHandEye` (eye-to-hand için gripper pozlarını ters çevirerek ver) → `T_base_cam`
4. `np.save("calib/T_base_cam.npy", T)` ve doğrula: işaretçinin kameradan hesaplanan konumu ile
   robotun bildiği konum arasındaki fark birkaç mm olmalı.

## Çalıştırma
```bash
python -m pip install pyrealsense2
python scripts/perceive_real.py --calib calib/T_base_cam.npy     # tespitleri logs/real.png'ye çizer
```

## Gerçek dünyada farklı olanlar
- **Işık:** HSV eşikleri ortama göre ayarlanmalı; OWLv2 (`--detector owl`) ışığa daha dayanıklı.
- **Derinlik gürültüsü:** RealSense'te ±2-5 mm; `measure()` yüzdelik/ortalama kullandığı için tolere eder.
- **Kavrama:** gerçek kolda gripper akımı/konumu ile temas algılama yapılır (`close_gripper` bunu döndürmeli).
