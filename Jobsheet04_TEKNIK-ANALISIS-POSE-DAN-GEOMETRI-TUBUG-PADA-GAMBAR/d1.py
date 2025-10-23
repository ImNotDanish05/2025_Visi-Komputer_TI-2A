import cv2, time

cap = cv2.VideoCapture(2)
frames = 0
t0 = time.time()
window_name = "Preview"

cv2.namedWindow(window_name)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frames += 1
    if time.time() - t0 >= 1.0:
        cv2.setWindowTitle(window_name, f"{window_name} (FPS ~ {frames})")
        frames = 0
        t0 = time.time()

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
