import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# --- Konfigurasi indeks mata kiri (berdasarkan landmark Mediapipe) ---
# Vertikal: (159, 145), Horizontal: (33, 133)
L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133

# --- Fungsi pembantu: menghitung jarak Euclidean antar dua titik ---
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --- Inisialisasi kamera ---
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# --- Inisialisasi FaceMeshDetector ---
detector = FaceMeshDetector(
    staticMode=False,        # False = deteksi terus-menerus setiap frame
    maxFaces=2,              # Maksimal wajah yang dideteksi
    minDetectionCon=0.5,     # Ambang batas kepercayaan deteksi
    minTrackCon=0.5          # Ambang batas kepercayaan pelacakan
)

# --- Variabel untuk mendeteksi kedipan ---
blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3     # Jumlah frame berturut-turut untuk dianggap "berkedip"
EYE_AR_THRESHOLD = 0.20         # Ambang EAR untuk menentukan mata tertutup
is_closed = False

# --- Loop utama ---
while True:
    ok, img = cap.read()
    if not ok:
        break

    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]  # Titik-titik wajah (468 koordinat)
        v = dist(face[L_TOP], face[L_BOTTOM])
        h = dist(face[L_LEFT], face[L_RIGHT])
        ear = v / (h + 1e-8)

        # Tampilkan nilai EAR
        cv2.putText(img, f"EAR(L): {ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # --- Logika kedipan ---
        if ear < EYE_AR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True
        else:
            closed_frames = 0
            is_closed = False

        # Tampilkan jumlah kedipan
        cv2.putText(img, f"Blink: {blink_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # --- Tampilkan hasil frame ---
    cv2.imshow("FaceMesh + EAR", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Bersihkan semua resources ---
cap.release()
cv2.destroyAllWindows()