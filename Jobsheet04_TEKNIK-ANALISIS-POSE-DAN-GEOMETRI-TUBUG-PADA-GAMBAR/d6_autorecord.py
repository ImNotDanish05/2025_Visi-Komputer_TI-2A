import cv2
import numpy as np
from collections import deque
from datetime import datetime
import os
from cvzone.PoseModule import PoseDetector

# ===================== PENGATURAN =====================
MODE = "squat"      # tekan 'm' untuk toggle ke "pushup"
KNEE_DOWN, KNEE_UP = 50, 100
DOWN_R, UP_R = 0.85, 1.00
SAMPLE_OK = 4

# buat folder rekaman
record_dir = "recordings"
os.makedirs(record_dir, exist_ok=True)

# nama file rekaman otomatis
filename = os.path.join(record_dir, f"{MODE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

# inisialisasi kamera
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# setup video recorder
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # bisa juga 'avc1' atau 'H264'
out          = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

print(f"[INFO] Merekam otomatis ke: {filename}")

# inisialisasi detektor pose
detector = PoseDetector(staticMode=False, modelComplexity=1,
                        enableSegmentation=False, detectionCon=0.5, trackCon=0.5)

count, state = 0, "up"
debounce = deque(maxlen=6)

# --------- fungsi bantu untuk push-up (pakai sudut siku) ---------
def elbow_angle(lm):
    sh = np.array(lm[11][1:3])
    el = np.array(lm[13][1:3])
    wr = np.array(lm[15][1:3])
    a = np.linalg.norm(el - wr)
    b = np.linalg.norm(sh - el)
    c = np.linalg.norm(sh - wr)
    angle = np.degrees(np.arccos((b**2 + a**2 - c**2) / (2*b*a + 1e-8)))
    return angle

# ===================== LOOP UTAMA =====================
while True:
    ok, img = cap.read()
    if not ok:
        break

    img = detector.findPose(img, draw=True)
    lmList, _ = detector.findPosition(img, draw=False)
    flag = None

    if lmList:
        if MODE == "squat":
            p1L, p2L, p3L = lmList[23][1:3], lmList[25][1:3], lmList[27][1:3]
            angL, _ = detector.findAngle(p1L, p2L, p3L, img)
            p1R, p2R, p3R = lmList[24][1:3], lmList[26][1:3], lmList[28][1:3]
            angR, _ = detector.findAngle(p1R, p2R, p3R, img)
            ang = 360 - ((angL + angR) / 2.0)
            if ang < KNEE_DOWN: flag = "down"
            elif ang > KNEE_UP: flag = "up"
            cv2.putText(img, f"Knee: {ang:5.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        else:  # push-up
            ang = elbow_angle(lmList)
            if ang < 90: flag = "down"
            elif ang > 150: flag = "up"
            cv2.putText(img, f"Elbow: {ang:5.1f}", (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        debounce.append(flag)
        if debounce.count("down") >= SAMPLE_OK and state == "up":
            state = "down"
        if debounce.count("up") >= SAMPLE_OK and state == "down":
            state = "up"
            count += 1

    # tampilkan & rekam
    cv2.putText(img, f"Mode: {MODE.upper()}  Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(img, f"State: {state}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    out.write(img)              # <--- simpan frame ke file
    cv2.imshow("Pose Counter", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('m'): MODE = "pushup" if MODE == "squat" else "squat"

# ===================== SELESAI =====================
cap.release()
out.release()
cv2.destroyAllWindows()
print("[INFO] Rekaman selesai dan disimpan ✅")
