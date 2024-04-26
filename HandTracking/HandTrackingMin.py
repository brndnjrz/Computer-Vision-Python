import cv2
import mediapipe as mp
import time

# initializes the webcam capture
# usually use 0 since 1 typically refers to the 2nd camera
cap = cv2.VideoCapture(1)

# initializes the Hand module from MediaPipe for hand detection
mpHands = mp.solutions.hands

# creates an instance of the Hand Class for detecting hands in webcam feed
hands = mpHands.Hands()

# initializes the drawing utilities from MediaPipe for drawing landmarks on the detected hands
mpDraw = mp.solutions.drawing_utils

# variables that stores previous time and current time for calculating frames per second
pTime = 0
cTime = 0

# loop continuously captures frames from the webcam, processes them to detect hands, and draws landmarks on the hands
while True:
    # captures a frame from the webcam and stores it in variable img
    success, img = cap.read()
    # converts img to RGB format bc 'hands' object only accepts RGB images
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # storing the results
    results = hands.process(imgRGB)
    # checks if hand is detected inside window frame and provide position if true
    # print(results.multi_hand_landmarks)

    # if statement iterates over each detected hand 'handLms' and each landmark 'lm' within each hand
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # landmark information will give us the x and y coordinates and ID number or index number total 21
            for id, lm in enumerate(handLms.landmark):
                # prints in console x, y, and z(id number)
                # print(id, lm)

                # extracts the x and y coordinates, calculates the position, and prints it to the console
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)   # include id to know which one its referring to

                # if statement to draw a circle around specific landmark
                # if id == 0:
                #     # 15 = size of dot
                #     cv2.circle(img, (cx, cy), 15, (192, 192, 192), cv2.FILLED)

                # to draw a circle on all of them instead of a specific one
                cv2.circle(img, (cx, cy), 15, (192, 192, 192), cv2.FILLED)

            # HAND_CONNECTIONS method will connect the dots with lines
            mpDraw.draw_landmarks(img,
                                  handLms,
                                  mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(255, 0, 0), circle_radius=6),   # configures dots on hands
                                  mpDraw.DrawingSpec(color=(192, 192, 192), thickness=5)    # configures lines on hands
                                  )

    cTime = time.time()
    # frames per second
    fps = 1/(cTime-pTime)
    pTime = cTime

    # calculates and displays the fps on the image
    cv2.putText(img,                     # img on which text will be drawn
                str(int(fps)),           # text displayed; converts frames per second to an integer, then to a string
                (10, 70),            # position where text will start; coordinates (x, y)
                cv2.FONT_HERSHEY_PLAIN,  # font type
                3,              # font scale
                (255, 0, 0),       # color of text; represented in BGR format
                3)              # thickness of text; higher value = thicker text

    # displays the image with the detected landmarks
    cv2.imshow("Image", img)

    # waits for a key press and continues the loop. if a key is pressed, the loop exits and program ends
    cv2.waitKey(1)
