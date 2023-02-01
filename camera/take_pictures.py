import cv2

cap = cv2.VideoCapture(0)
i = 0
while (1):
    ret, frame = cap.read()
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('s'):
        """按下s拍照"""
        cv2.imwrite('biaoding/' + str(i) + '.jpg', frame)
        i += 1
    cv2.imshow("capture", frame)
cap.release()
cv2.destroyAllWindows()