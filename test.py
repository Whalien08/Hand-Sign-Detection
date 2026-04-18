import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from cvzone.ClassificationModule import Classifier
import tensorflow 

cap=cv2.VideoCapture(0)
det=HandDetector(maxHands=1)

classifier= Classifier("model/my_asl_model.h5","model/labels.txt")
offset=20
imgSize=300

folder="Data/C"
counter=0

labels = labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

while True:
    success, img = cap.read()
    #img = cv2.flip(img, 1)
    if not success: break
    
    imgOutput = img.copy()
    hands, img = det.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # 1. Initialize imgWhite here so it's fresh for every detection
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Draw the UI on imgOutput
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 100, y - offset), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), 
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            # 2. Only show these windows if a hand is actually being processed
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # 3. Always show the main webcam feed
    cv2.imshow("Image", imgOutput)
    
    # Press 'q' to quit nicely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()