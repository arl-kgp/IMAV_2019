import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("outpy.avi")

def nothing(x):
    pass

cv2.namedWindow("frame")
cv2.createTrackbar("h_low", "frame", 98, 255, nothing)
cv2.createTrackbar("s_low", "frame", 79, 255, nothing)
cv2.createTrackbar("v_low", "frame", 78, 255, nothing)
cv2.createTrackbar("h_high", "frame", 128, 255, nothing)
cv2.createTrackbar("s_high", "frame", 245, 255, nothing)
cv2.createTrackbar("v_high", "frame", 169, 255, nothing)
font = cv2.FONT_HERSHEY_COMPLEX
kernel = np.ones((5,5), np.uint8)
while(1):

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

    row_avg = np.sum(res,1)/frame.shape[1]

    row_avg = row_avg[:,0]

    n_max = np.amax(row_avg)
    if (n_max>80):
        k = row_avg.argmax()
        k_val = np.amax(row_avg)
        row_avg[k-20:k+20] = 0
        t = row_avg.argmax()
        t_val = np.amax(row_avg)
        row_avg[t-20:t+20] = 0
        row_avg[k] = k_val
        row_avg[t] = t_val


        # print("max value is {}".format(n_max))
    #     # print(k)

    #     return 1, np.amin(np.where(row_avg>=80))

    # else:
    #     return 0,0



        f = np.where(row_avg>=80)
        l = np.amax(f)
        print(l)

        cv2.line(frame,(0,l),(frame.shape[1],l),(255,255,255),5)


        # condition = (f<(np.amax(f)-10))
        # l = f[]



        print("indices are {}".format(f))
        print("hoohah")
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()