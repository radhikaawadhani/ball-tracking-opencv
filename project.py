import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# Initialize the Video
cap = cv2.VideoCapture(0)  # Change to the correct video source or use 'Videos/vid (4).mp4' for a video file

# Create the color Finder object
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 150, 'vmin': 150, 'hmax': 14, 'smax': 255, 'vmax': 255}

# Variables
posListX, posListY = [], []
xList = [item for item in range(0, 1300)]
prediction = False

while True:
    # Grab the image
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    img = img[0:900, :]  # Crop the image if needed

    # Find the Color Ball
    imgColor, mask = myColorFinder.update(img, hsvVals)  # Get the color mask

    # Find location of the Ball/draw boundary of ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    if len(posListX) >= 3:  # Ensure there are enough points for fitting
            # Polynomial Regression y = Ax^2 + Bx + C
            A, B, C = np.polyfit(posListX, posListY, 2)

            # Plotting the trajectory
            for i, (posX, posY) in enumerate(zip(posListX, posListY)):
                pos = (posX, posY)
                cv2.circle(imgContours, pos, 10, (0, 255, 0), cv2.FILLED)
                if i > 0:
                    cv2.line(imgContours, (posListX[i - 1], posListY[i - 1]), pos, (0, 255, 0), 5)

            # Predict the trajectory
            for x in xList:
                y = int(A * x ** 2 + B * x + C)
                cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

            # Prediction logic
            if len(posListX) >= 10:
                a = A
                b = B
                c = C - 590  # Adjust for the hoop height

                # Calculate the intersection with y = 590
                discriminant = b ** 2 - 4 * a * c
                if discriminant >= 0:  # Only proceed if the result is real
                    x_prediction = int((-b + math.sqrt(discriminant)) / (2 * a))
                    prediction = 330 < x_prediction < 430  # Adjust for hoop range
            #for basketball prediction

        #if prediction:
            #cvzone.putTextRect(imgContours, "Basket", (50, 150),
                               #scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        #else:
            #cvzone.putTextRect(imgContours, "No Basket", (50, 150),
                               #scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

    # Display the processed image
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    cv2.imshow("ImageColor", imgContours)

    key = cv2.waitKey(100)
    if key == ord("s"):  # Press 's' to stop and restart
        posListX, posListY = [], []

# Release the capture and close windows after the loop ends
cap.release()
cv2.destroyAllWindows()

# Second part (for video file playback and prediction):
cap = cv2.VideoCapture(0)  # Change to the correct video file path
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 124, 'vmin': 13, 'hmax': 24, 'smax': 255, 'vmax': 255}

posListX = []
posListY = []

listX = [item for item in range(0, 1300)]
start = True
prediction = False

while True:
    if start:
        if len(posListX) == 10: start = False
        success, img = cap.read()
        if not success:
            print("Failed to read video frame")
            break

        img = img[0:900, :]
        imgPrediction = img.copy()
        imgResult = img.copy()
        imgBall, mask = myColorFinder.update(img, hsvVals)
        imgCon, contours = cvzone.findContours(img, mask, 200)
        if contours:
            posListX.append(contours[0]['center'][0])
            posListY.append(contours[0]['center'][1])

        if posListX:
            if len(posListX) < 18:
                coff = np.polyfit(posListX, posListY, 2)
            for i, (posX, posY) in enumerate(zip(posListX, posListY)):
                pos = (posX, posY)
                cv2.circle(imgCon, pos, 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(imgResult, pos, 10, (0, 255, 0), cv2.FILLED)

                if i == 0:
                    cv2.line(imgCon, pos, pos, (0, 255, 0), 2)
                    cv2.line(imgResult, pos, pos, (0, 255, 0), 2)
                else:
                    cv2.line(imgCon, (posListX[i - 1], posListY[i - 1]), pos, (0, 255, 0), 2)
                    cv2.line(imgResult, (posListX[i - 1], posListY[i - 1]), pos, (0, 255, 0), 2)

            for x in listX:
                y = int(coff[0] * x ** 2 + coff[1] * x + coff[2])
                cv2.circle(imgPrediction, (x, y), 2, (255, 0, 255), cv2.FILLED)
                cv2.circle(imgResult, (x, y), 2, (255, 0, 255), cv2.FILLED)

            # Predict
            if len(posListX) < 10:
                a, b, c = coff
                c = c - 593
                x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))
                prediction = 300 < x < 430

            if prediction:
                cvzone.putTextRect(imgResult, "Basket", (50, 150), colorR=(0, 200, 0),
                                   scale=5, thickness=10, offset=20)
            else:
                cvzone.putTextRect(imgResult, "No Basket", (50, 150), colorR=(0, 0, 200),
                                   scale=5, thickness=10, offset=20)

        cv2.line(imgCon, (330, 593), (430, 593), (255, 0, 255), 10)
        imgResult = cv2.resize(imgResult, (0, 0), None, 0.7, 0.7)

        cv2.imshow("imgCon", imgResult)

    key = cv2.waitKey(100)
    if key == ord("s"):
        start = True

# Release the capture and close windows after the loop ends
cap.release()
cv2.destroyAllWindows()