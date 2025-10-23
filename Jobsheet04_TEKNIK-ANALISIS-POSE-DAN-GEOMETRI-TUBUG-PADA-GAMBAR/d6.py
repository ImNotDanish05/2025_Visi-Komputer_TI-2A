import cv2
import numpy as np
from collections import deque
from cvzone.PoseModule import PoseDetector

# Mode awal: squat atau push-up
MODE = "squat"  # tekan 'm' untuk toggle ke "pushup"
KNEE_DOWN, KNEE_UP = 50, 100  # ambang squat (derajat)
DOWN_R, UP_R = 0.85, 1.00     # ambang push-up (rasio)
SAMPLE_OK = 4                 # minimal frame konsisten sebelum ganti state

# Gunakan kamera index 2
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi pose detector
detector = PoseDetector(
    staticMode=False,
    modelComplexity=1,
    enableSegmentation=False,
    detectionCon=0.5,
    trackCon=0.5
)

count, state = 0, "up"
debounce = deque(maxlen=6)

# --- Fungsi rasio push-up ---
def ratio_pushup(lm):
    # gunakan titik kiri: 11=shoulderL, 15=wristL, 23=hipL
    sh = np.array(lm[11][1:3])
    wr = np.array(lm[15][1:3])
    hp = np.array(lm[23][1:3])
    return np.linalg.norm(sh - wr) / (np.linalg.norm(sh - hp) + 1e-8)

def elbow_angle(lm):
    # titik kiri: 11=shoulderL, 13=elbowL, 15=wristL
    sh = np.array(lm[11][1:3])
    el = np.array(lm[13][1:3])
    wr = np.array(lm[15][1:3])
    # hitung sudut siku (menggunakan rumus vektor)
    a = np.linalg.norm(el - wr)
    b = np.linalg.norm(sh - el)
    c = np.linalg.norm(sh - wr)
    angle = np.degrees(np.arccos((b**2 + a**2 - c**2) / (2*b*a + 1e-8)))
    return angle


# --- Loop utama ---
while True:
    ok, img = cap.read()
    if not ok:
        break

    img = detector.findPose(img, draw=True)
    lmList, _ = detector.findPosition(img, draw=False)  # [(id,x,y,z,vis), ...]
    flag = None

    if lmList:
        if MODE == "squat":
            # Ambil koordinat dari lmList
            p1L = lmList[23][1:3]  # Hip kiri
            p2L = lmList[25][1:3]  # Knee kiri
            p3L = lmList[27][1:3]  # Ankle kiri
            angL, _ = detector.findAngle(p1L, p2L, p3L, img)

            p1R = lmList[24][1:3]  # Hip kanan
            p2R = lmList[26][1:3]  # Knee kanan
            p3R = lmList[28][1:3]  # Ankle kanan
            angR, _ = detector.findAngle(p1R, p2R, p3R, img)

            ang = 360 - ((angL + angR) / 2.0)
            if ang < KNEE_DOWN:
                flag = "down"
            elif ang > KNEE_UP:
                flag = "up"

            cv2.putText(img, f"Knee: {ang:5.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:  # MODE == "pushup"
            ang = elbow_angle(lmList)
            if ang < 90:
                flag = "down"
            elif ang > 150:
                flag = "up"

            cv2.putText(img, f"Elbow: {ang:5.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Debounce logic
        debounce.append(flag)
        if debounce.count("down") >= SAMPLE_OK and state == "up":
            state = "down"
        if debounce.count("up") >= SAMPLE_OK and state == "down":
            state = "up"
            count += 1

    # --- Tampilkan info di layar ---
    cv2.putText(img, f"Mode: {MODE.upper()}  Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"State: {state}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Pose Counter", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('m'):
        MODE = "pushup" if MODE == "squat" else "squat"

# --- Bersihkan sumber daya ---
cap.release()
cv2.destroyAllWindows()
