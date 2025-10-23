import cv2
import numpy as np
import os
import time
from cvzone.HandTrackingModule import HandDetector

# ====== Pengaturan dasar ======
COOLDOWN = 1.0  # waktu tunggu (detik) sebelum ambil foto berikutnya
OUTPUT_DIR = "output"

# ====== Fungsi bantu ======
def ensure_dir(path):
    """Buat folder jika belum ada."""
    if not os.path.exists(path):
        os.makedirs(path)

def dist(a, b):
    """Hitung jarak Euclidean antara dua titik."""
    return np.linalg.norm(np.array(a) - np.array(b))

def classify_gesture(hand):
    """Klasifikasi gestur berdasarkan landmark tangan."""
    lm = hand["lmList"]
    wrist = np.array(lm[0][:2])
    thumb_tip = np.array(lm[4][:2])
    index_tip = np.array(lm[8][:2])
    middle_tip = np.array(lm[12][:2])
    ring_tip = np.array(lm[16][:2])
    pinky_tip = np.array(lm[20][:2])

    r_mean = np.mean([
        dist(index_tip, wrist),
        dist(middle_tip, wrist),
        dist(ring_tip, wrist),
        dist(pinky_tip, wrist),
        dist(thumb_tip, wrist)
    ])

    if dist(thumb_tip, index_tip) < 35:
        return "OK"

    if (thumb_tip[1] < wrist[1] - 40) and (dist(thumb_tip, wrist) > 0.8 * dist(index_tip, wrist)):
        return "THUMBS_UP"

    if r_mean < 120:
        return "ROCK"

    if r_mean > 200:
        return "PAPER"

    if (
        dist(index_tip, wrist) > 180 and
        dist(middle_tip, wrist) > 180 and
        dist(ring_tip, wrist) < 160 and
        dist(pinky_tip, wrist) < 160
    ):
        return "SCISSORS"

    return "UNKNOWN"

# ====== Inisialisasi kamera ======
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# ====== Inisialisasi detektor tangan ======
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

# ====== Variabel status ======
last_capture_time = 0
last_label = None

# Pastikan folder output utama ada
ensure_dir(OUTPUT_DIR)

# ====== Loop utama ======
while True:
    ok, img = cap.read()
    if not ok:
        break

    hands, img = detector.findHands(img, draw=True, flipType=True)
    label = "UNKNOWN"

    if hands:
        label = classify_gesture(hands[0])
        cv2.putText(img, f"Gesture: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # --- Screenshot otomatis ---
        now = time.time()
        if (now - last_capture_time) >= COOLDOWN:
            # Pastikan folder class ada
            gesture_dir = os.path.join(OUTPUT_DIR, label)
            ensure_dir(gesture_dir)

            # Buat nama file unik berdasarkan waktu
            filename = os.path.join(
                gesture_dir,
                f"{label}_{int(time.time())}.jpg"
            )

            cv2.imwrite(filename, img)
            print(f"[INFO] Screenshot disimpan di: {filename}")

            last_capture_time = now
            last_label = label

    # Tampilkan hasil
    cv2.imshow("Hand Gestures (cvzone)", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====== Bersihkan sumber daya ======
cap.release()
cv2.destroyAllWindows()
