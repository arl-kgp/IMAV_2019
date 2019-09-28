import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("outpy.avi")

def nothing(x):
    pass

cv2.namedWindow("frame")
cv2.createTrackbar("h_low", "frame", 0, 255, nothing)
cv2.createTrackbar("s_low", "frame", 0, 255, nothing)
cv2.createTrackbar("v_low", "frame", 0, 255, nothing)
cv2.createTrackbar("h_high", "frame",255, 255, nothing)
cv2.createTrackbar("s_high", "frame", 255, 255, nothing)
cv2.createTrackbar("v_high", "frame", 255, 255, nothing)
font = cv2.FONT_HERSHEY_COMPLEX
kernel = np.ones((5,5), np.uint8)
first = 0
while(1):

    k = cv2.waitKey(5) & 0xFF

    if(first == 0):
        _,frame = cap.read()
        first = 1

    else:

        if(k == ord("m")):
            # Take each frame
            _, frame = cap.read()

        h_low = cv2.getTrackbarPos("h_low","frame")
        s_low = cv2.getTrackbarPos("s_low","frame")
        v_low = cv2.getTrackbarPos("v_low","frame")

        h_high = cv2.getTrackbarPos("h_high","frame")
        s_high = cv2.getTrackbarPos("s_high","frame")
        v_high = cv2.getTrackbarPos("v_high","frame")    

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([h_low,s_low,v_low])
        upper_blue = np.array([h_high,s_high,v_high])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)

        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)

        if k == 27:
            break

cv2.destroyAllWindows()