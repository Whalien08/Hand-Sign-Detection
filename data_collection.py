import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap=cv2.VideoCapture(0)
det=HandDetector(maxHands=1)
offset=20
imgSize=300

folder="Data/A"
counter=0

while True:
    success, img = cap.read()
    hands, img = det.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Reset imgWhite every time a hand is detected so it's clean
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                # Case: Tall Hand (fix height to 300)
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                # Case: Wide Hand (fix width to 300)
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
            
    cv2.imshow("Image", img)
    key= cv2.waitKey(1)

    if key == ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)