import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while  True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0),  # Dots (invisible)
                mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3)  # Connections (radium green)
            )

    cv2.imshow("Image",img)
    cv2.waitKey(1)