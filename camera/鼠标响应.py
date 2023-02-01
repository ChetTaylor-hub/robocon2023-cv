import cv2
import numpy as np
from util import Pose_estimation
from p3p import P3P

img = cv2.imread('Adirondack-perfect/im0.png')
img = cv2.resize(img, [int(2880 / 4), int(1988 / 4)])
pose_estimation = Pose_estimation()
p4p = P3P(np.array([[4161.221, 0, 1445.577], [0, 4161.221, 984.686], [0, 0, 1]]) / 4,
          np.array([[-0.1065, 0.0793, -0.0002, -8.9263e-06, -0.0161]]))


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pitch, yaw = pose_estimation.pixelToangle(x, y)
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.putText(img, str(pitch), (x, y - 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.putText(img, str(yaw), (x, y - 30), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while (1):
    cv2.imshow("image", img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()
