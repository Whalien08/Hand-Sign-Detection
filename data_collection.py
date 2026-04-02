import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap=cv2.VideoCapture(0)
det=HandDetector(maxHands=1)
offset=20
imgSize=300

while True:
    success, img = cap.read()
    hands, img = det.findHands(img)
    if hands:
        hand =hands[0]
        x,y,w,h = hand['bbox']

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size>0:
            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
            h_crop, w_crop, _ = imgCrop.shape
            h_final = min(h_crop, imgSize)
            w_final = min(w_crop, imgSize)

            imgWhite[0:h_final, 0:w_final] = imgCrop[:h_final, :w_final]
        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)
    cv2.imshow("Image",img)
    cv2.waitKey(1) 