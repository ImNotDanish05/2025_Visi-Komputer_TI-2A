import cv2
from cvzone.PoseModule import PoseDetector
cap = cv2.VideoCapture(2)  # => Diubah sesuai webcam selection

detector = PoseDetector()

while True:
    # Membaca video dari webcam
    success, frame = cap.read()

    frame = detector.findPose(frame, draw=True)
    lmList, bboxInfo = detector.findPosition(frame,
                                            draw=True, 
                                            bboxWithHands=False
                                            )
    
    if (lmList):
        length, frame, info = detector.findDistance(lmList[11][0:2],
                                                    lmList[15][0:2], 
                                                    img=frame,
                                                    color=(255, 0, 255), # => BGR (Blue, Green, Red)
                                                    scale=10
                                                    )
        print(length)

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()