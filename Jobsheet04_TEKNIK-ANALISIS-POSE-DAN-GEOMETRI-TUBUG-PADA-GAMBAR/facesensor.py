import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# Inisialisasi detector
detector = FaceMeshDetector(maxFaces=1)
cap = cv2.VideoCapture(2)

# Load gambar overlay
overlay = cv2.imread('face.png', cv2.IMREAD_UNCHANGED)  # support alpha channel

while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]

        # Dapatkan koordinat wajah
        x_list = [p[0] for p in face]
        y_list = [p[1] for p in face]
        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)

        # Hitung lebar dan tinggi wajah
        w, h = x_max - x_min, y_max - y_min

        # Resize overlay sesuai ukuran wajah
        resized_overlay = cv2.resize(overlay, (w, h))

        # Buat ROI (Region of Interest)
        roi = img[y_min:y_min+h, x_min:x_min+w]

        # Cek apakah overlay punya alpha channel
        # --- Tempel overlay ke wajah ---
        if resized_overlay.shape[2] == 4:
            alpha = resized_overlay[:, :, 3] / 255.0
            h, w = resized_overlay.shape[:2]

            # Pastikan tidak keluar dari frame
            y1, y2 = max(0, y_min), min(img.shape[0], y_min + h)
            x1, x2 = max(0, x_min), min(img.shape[1], x_min + w)

            overlay_crop = resized_overlay[0:(y2 - y1), 0:(x2 - x1)]
            alpha = overlay_crop[:, :, 3] / 255.0
            for c in range(3):
                img[y1:y2, x1:x2, c] = (
                    (1 - alpha) * img[y1:y2, x1:x2, c] +
                    alpha * overlay_crop[:, :, c]
                )
        else:
            # Versi tanpa transparansi
            h, w = resized_overlay.shape[:2]
            y1, y2 = max(0, y_min), min(img.shape[0], y_min + h)
            x1, x2 = max(0, x_min), min(img.shape[1], x_min + w)
            img[y1:y2, x1:x2] = resized_overlay[0:(y2 - y1), 0:(x2 - x1)]


        # Tempel hasil kembali ke gambar utama
        img[y_min:y_min+h, x_min:x_min+w] = roi

    cv2.imshow("Face Cover Filter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
