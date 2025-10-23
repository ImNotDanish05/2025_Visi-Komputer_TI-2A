import cv2

# Coba cek dari 0 sampai 10 (bisa disesuaikan)
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Kamera ditemukan di index: {i}")
        cap.release()
    else:
        print(f"❌ Tidak ada kamera di index: {i}")
