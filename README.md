<p align="center"><a href="https://www.youtube.com/@ImNotDanish05"><img src="danish05.png" width="400" alt="Created by ImNotDanish05"></a></p>

<p align="center">
<a href="https://github.com/ImNotDanish05"><img src="https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github" alt="GitHub"></a>
<a href="https://www.youtube.com/@ImNotDanish05"><img src="https://img.shields.io/badge/YouTube-Channel-FF0000?style=for-the-badge&logo=youtube" alt="YouTube"></a>
</p>

# 2025_Visi-Komputer_TI-2A

Materi dan tugas mata kuliah Visi Komputer (TI-2A, 2025). Repo ini berisi jobsheet praktik dan tugas yang menutup dasar klasifikasi gambar, regresi berbasis citra, analisis pose/geometri tubuh secara real-time, serta segmentasi (MediaPipe dan SAM2).

## Isi repositori
- `Jobsheet02_KLASIFIKASI-GAMBAR.ipynb` — pengantar klasifikasi gambar dengan dataset sederhana (preview sampel, pemodelan dasar), disertai contoh image assets di `Asset02_KLASIFIKASI-GAMBAR/`.
- `Tugas02_KLASIFIKASI-GAMBAR.ipynb` — klasifikasi MNIST dengan TensorFlow/Keras, mulai dari loading data hingga evaluasi dan eksperimen sederhana.
- `Jobsheet03_TEKNIK-REGRESI-GAMBAR.ipynb` — tiga praktik regresi dari citra: prediksi radius lingkaran dari data sintetis, prediksi umur pada UTKFace, dan penilaian popularitas hewan peliharaan.
- `Tugas03_TEKNIK-REGRESI-GAMBAR.ipynb` — lanjutan regresi dengan variasi model (ResNet50, EfficientNetB3, hingga fitur non-visual) pada set kasus Jobsheet 03.
- `Jobsheet04_TEKNIK-ANALISIS-POSE-DAN-GEOMETRI-TUBUG-PADA-GAMBAR/` — skrip real-time OpenCV + cvzone/MediaPipe untuk deteksi pose, Face Mesh (EAR counter), hand detection, gesture classifier, squat/push-up counter, dan face overlay. Lihat README di folder tersebut untuk detail dan hotkeys.
- `Jobsheet05_Segmentasi-Gambar/` — demo segmentasi menggunakan MediaPipe Tasks (selfie/hair segmentation, background removal/replace) serta eksperimen SAM2 hair segmentation dan web app Flask (`sam2_web_py/`) untuk pemrosesan kamera ponsel via Wi-Fi.

## Cara pakai cepat
- Notebook: jalankan langsung di Google Colab (badge di sel pertama) atau lokal dengan Python 3.9+.
- Skrip real-time (Jobsheet04): pastikan webcam terhubung, instal dependensi utama `opencv-python numpy cvzone mediapipe`, lalu jalankan misalnya `python Jobsheet04_TEKNIK-ANALISIS-POSE-DAN-GEOMETRI-TUBUG-PADA-GAMBAR/d1.py`. Banyak skrip memakai `VideoCapture(2)`; ubah ke index kamera Anda jika perlu.
- Segmentasi (Jobsheet05): instal `mediapipe opencv-python numpy requests` lalu jalankan `selfie_segmentation.py`, `hair_segmentation.py`, `background_removal.py`, atau `background_replace.py`. Model akan otomatis diunduh ke folder `models/` saat pertama dipakai.
- SAM2 web app: `cd Jobsheet05_Segmentasi-Gambar/sam2_web_py && pip install -r requirements.txt && python app.py`, buka `http://<ip-laptop>:8000` dari ponsel di jaringan yang sama, lalu tombol Capture akan mengirim frame ke backend SAM2.

## Catatan
- Beberapa skrip membutuhkan koneksi internet saat pertama kali untuk mengunduh model (MediaPipe TFLite atau checkpoint SAM2 ke `.checkpoints/`).
- Gunakan virtual environment agar dependensi proyek tidak bentrok.
- Tekan `q` untuk keluar dari jendela video pada skrip real-time; pada counter squat/push-up gunakan `m` untuk memindah mode.
