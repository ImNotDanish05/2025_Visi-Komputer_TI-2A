import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Fungsi untuk menghitung jarak Euclidean antara dua titik
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Fungsi untuk mengklasifikasi gestur tangan
def classify_gesture(hand):
    # hand["lmList"] berisi 21 titik (x,y,z)
    lm = hand["lmList"]

    wrist = np.array(lm[0][:2])
    thumb_tip = np.array(lm[4][:2])
    index_tip = np.array(lm[8][:2])
    middle_tip = np.array(lm[12][:2])
    ring_tip = np.array(lm[16][:2])
    pinky_tip = np.array(lm[20][:2])

    # Hitung rata-rata jarak ujung jari ke pergelangan tangan
    r_mean = np.mean([
        dist(index_tip, wrist),
        dist(middle_tip, wrist),
        dist(ring_tip, wrist),
        dist(pinky_tip, wrist),
        dist(thumb_tip, wrist)
    ])

    # Aturan deteksi gestur
    if dist(thumb_tip, index_tip) < 35:
        return "OK"

    # Thumbs Up: ibu jari tinggi (y lebih kecil) dan jauh dari pergelangan
    if (thumb_tip[1] < wrist[1] - 40) and (dist(thumb_tip, wrist) > 0.8 * dist(index_tip, wrist)):
        return "THUMBS_UP"

    # ROCK: rata-rata jarak jari kecil → kepalan
    if r_mean < 120:
        return "ROCK"

    # PAPER: rata-rata jarak besar → tangan terbuka
    if r_mean > 200:
        return "PAPER"

    # SCISSORS: dua jari panjang, dua jari pendek
    if (
        dist(index_tip, wrist) > 180 and
        dist(middle_tip, wrist) > 180 and
        dist(ring_tip, wrist) < 160 and
        dist(pinky_tip, wrist) < 160
    ):
        return "SCISSORS"

    return "UNKNOWN"


# Inisialisasi kamera
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi detektor tangan
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

# Loop utama
while True:
    ok, img = cap.read()
    if not ok:
        break

    # Deteksi tangan
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        label = classify_gesture(hands[0])
        cv2.putText(img, f"Gesture: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Tampilkan jendela hasil
    cv2.imshow("Hand Gestures (cvzone)", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
