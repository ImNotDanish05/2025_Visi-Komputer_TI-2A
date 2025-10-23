# Jobsheet 04 — Teknik Analisis Pose dan Geometri Tubuh pada Gambar

Dibuat oleh: ImNotDanish05

Dokumen ini merangkum seluruh skrip dalam folder, menjelaskan fungsi masing-masing, dependensi yang dibutuhkan, serta cara menjalankannya. Proyek ini berfokus pada analisis pose, wajah, dan tangan secara real‑time menggunakan OpenCV, cvzone (MediaPipe), dan NumPy.

---

## Ringkasan

- Realtime capture dari webcam dan preview FPS.
- Deteksi pose dan pengukuran jarak/rasio landmark tubuh.
- Deteksi Face Mesh dan perhitungan EAR (Eye Aspect Ratio) untuk hitung kedipan.
- Deteksi tangan, hitung jumlah jari, dan klasifikasi gestur sederhana.
- Counter squat/push‑up berbasis sudut lutut/rasio posisi tubuh.
- Face overlay filter menggunakan gambar `face.png` (mendukung alpha channel).

---

## Prasyarat

- Python 3.8+
- Sistem telah memiliki akses ke kamera (webcam) yang tidak dipakai aplikasi lain.

### Dependensi Python

Instal dengan pip (disarankan dalam virtual environment):

```bash
pip install opencv-python numpy cvzone mediapipe
```

Catatan:
- Pada beberapa OS, paket `opencv-python` dapat diganti `opencv-contrib-python` jika diperlukan.
- `cvzone` sudah menyertakan wrapper untuk modul MediaPipe.

---

## Struktur Proyek

- `camerachecker.py`: Enumerasi index kamera yang tersedia.
- `testcamera.py`: Preview webcam + FPS (setara dengan `d1.py`).
- `test.py`: Cek modul-modul yang tersedia di paket `cvzone`.
- `facesensor.py`: Face Mesh + overlay gambar `face.png` di area wajah.
- `d1.py`: Preview webcam + FPS, tekan `q` untuk keluar.
- `d2.py`: Deteksi pose dan hitung jarak antar landmark (contoh: bahu–pergelangan tangan kiri).
- `d3.py`: Face Mesh + perhitungan EAR (mata kiri) untuk hitung jumlah kedipan.
- `d4.py`: Deteksi tangan dan hitung jumlah jari yang terangkat.
- `d5.py`: Klasifikasi gestur tangan (OK, THUMBS_UP, ROCK, PAPER, SCISSORS, UNKNOWN) berbasis heuristik jarak.
- `d6.py`: Counter squat/push‑up dengan debounce state untuk akurasi.
- `face.png`: Gambar overlay untuk `facesensor.py` (sebaiknya PNG dengan alpha).

---

## Cara Menentukan Index Kamera

Banyak skrip menggunakan `cv2.VideoCapture(2)`. Jika kamera Anda tidak berada di index `2`, jalankan:

```bash
python camerachecker.py
```

Pilih index yang terbuka (tersedia), lalu ubah angka pada `cv2.VideoCapture(...)` di skrip terkait bila perlu.

---

## Panduan Pemakaian per Skrip

Semua skrip dapat dihentikan dengan menekan tombol `q` pada jendela video.

### 1) Preview Webcam + FPS

- File: `d1.py`, `testcamera.py`
- Jalankan:
  ```bash
  python d1.py
  # atau
  python testcamera.py
  ```
- Fitur: Menampilkan feed kamera dengan estimasi FPS pada title bar.

### 2) Pose: Jarak Antar Landmark

- File: `d2.py`
- Jalankan:
  ```bash
  python d2.py
  ```
- Fitur: Deteksi pose (cvzone PoseDetector) dan menghitung jarak pixel antara landmark 11 (bahu kiri) dan 15 (pergelangan tangan kiri). Nilai jarak dicetak ke terminal.

### 3) Face Mesh + Kedipan (EAR)

- File: `d3.py`
- Jalankan:
  ```bash
  python d3.py
  ```
- Fitur: Deteksi Face Mesh dan menghitung EAR untuk mata kiri menggunakan titik 159/145 (vertikal) dan 33/133 (horizontal). Jika EAR turun di bawah ambang beberapa frame berturut-turut, dihitung sebagai satu kedipan.
- Parameter penting di dalam skrip:
  - `EYE_AR_THRESHOLD = 0.20`
  - `CLOSED_FRAMES_THRESHOLD = 3`

### 4) Deteksi Tangan + Hitung Jari

- File: `d4.py`
- Jalankan:
  ```bash
  python d4.py
  ```
- Fitur: Mendeteksi satu tangan dan menghitung jumlah jari terangkat menggunakan `cvzone.HandTrackingModule.HandDetector`. Tampilan list 0/1 per jari dan totalnya.

### 5) Klasifikasi Gestur Tangan (Heuristik)

- File: `d5.py`
- Jalankan:
  ```bash
  python d5.py
  ```
- Fitur: Mengklasifikasi gestur sederhana berdasarkan jarak landmark ke pergelangan:
  - `OK` (ibu jari dekat telunjuk)
  - `THUMBS_UP` (ibu jari mengarah ke atas dan relatif jauh dari pergelangan)
  - `ROCK` (kepalan, rata-rata jarak ujung jari ke pergelangan kecil)
  - `PAPER` (tangan terbuka, rata-rata jarak besar)
  - `SCISSORS` (dua jari panjang, dua jari pendek)
  - `UNKNOWN` (tidak masuk kriteria)
- Ambang jarak pada skrip dapat disesuaikan sesuai resolusi kamera dan ukuran tangan di frame.

### 6) Counter Squat / Push‑Up (Toggle)

- File: `d6.py`
- Jalankan:
  ```bash
  python d6.py
  ```
- Fitur: Menghitung repetisi `squat` atau `pushup` dengan debounce untuk menstabilkan state.
  - Toggle mode: tekan `m` (default `squat` → `pushup` → `squat` …)
  - `squat`: memakai sudut lutut kiri/kanan (hip–knee–ankle). Rata-rata sudut dikonversi ke kisaran dan dibandingkan dengan ambang.
  - `pushup`: memakai rasio jarak `shoulder–wrist` terhadap `shoulder–hip`.
- Parameter utama (dalam skrip):
  - `KNEE_DOWN = 50`, `KNEE_UP = 100` (derajat)
  - `DOWN_R = 0.85`, `UP_R = 1.00` (rasio)
  - `SAMPLE_OK = 4` (berapa banyak frame konsisten untuk mengganti state)

### 7) Face Overlay Filter

- File: `facesensor.py`
- Prasyarat: pastikan `face.png` tersedia di folder yang sama.
- Jalankan:
  ```bash
  python facesensor.py
  ```
- Fitur: Mendeteksi Face Mesh lalu menempelkan gambar `face.png` mengikuti bounding wajah. Jika PNG memiliki alpha, akan dihitung blending per‑channel.
- Tips:
  - Ukuran/proporsi overlay mengikuti bounding wajah; siapkan PNG transparan yang sesuai.
  - Jika overlay tampak tidak rata/terpotong, cek batas ROI yang dipotong agar tidak keluar dari frame.

### 8) Cek Modul cvzone

- File: `test.py`
- Jalankan:
  ```bash
  python test.py
  ```
- Fitur: Menampilkan daftar submodul dalam paket `cvzone` yang terpasang.

---

## Troubleshooting

- Kamera tidak terbuka:
  - Pastikan index kamera benar (gunakan `python camerachecker.py`).
  - Tutup aplikasi lain yang memakai webcam (Teams/Zoom/OBS, dsb.).
  - Cek permission kamera pada OS.
- Performa lambat:
  - Turunkan resolusi capture (`cap.set(cv2.CAP_PROP_FRAME_WIDTH, ...)` dan `...HEIGHT`).
  - Gunakan mode `maxFaces=1` atau `maxHands=1` jika tidak perlu mendeteksi banyak objek.
- Gestur tidak akurat (`d5.py`):
  - Sesuaikan ambang jarak untuk resolusi kamera Anda.
  - Pastikan tangan cukup besar di frame dan pencahayaan memadai.
- Overlay wajah kurang pas (`facesensor.py`):
  - Coba PNG dengan ukuran berbeda atau modifikasi perhitungan skala/penempatan.

---

## Catatan Teknis

- Tombol umum: `q` untuk keluar dari semua skrip; `m` untuk toggle mode pada `d6.py`.
- Hampir semua skrip menggunakan `cv2.VideoCapture(2)`; sesuaikan index kamera Anda.
- Beberapa karakter cetak pada output Windows terminal bisa tampil aneh (misalnya simbol/emoji) — ini tidak memengaruhi fungsi skrip.

---

## Kredit

- OpenCV — Computer Vision library
- MediaPipe (melalui cvzone) — Pose/Face/Hands landmark
- cvzone — Wrapper utilitas untuk MediaPipe + OpenCV

Dikompilasi dan didokumentasikan oleh ImNotDanish05.

