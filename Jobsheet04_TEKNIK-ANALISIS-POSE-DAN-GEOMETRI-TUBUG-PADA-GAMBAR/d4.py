import cv2
from cvzone.HandTrackingModule import HandDetector

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
    hands, img = detector.findHands(img, draw=True, flipType=True)  # flipType=True untuk mirror UI
    if hands:
        hand = hands[0]  # dict berisi "lmList", "bbox", dll.
        fingers = detector.fingersUp(hand)  # list panjang 5 berisi 0/1
        count = sum(fingers)

        # Tampilkan hasil
        cv2.putText(img, f"Fingers: {count} {fingers}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Tampilkan jendela
    cv2.imshow("Hands + Fingers", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
