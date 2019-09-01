import cv2
import numpy as np

class detect(object):

    def __init__(self):
        self.h_low = 86
        self.h_high = 121
        self.s_low = 78
        self.s_high = 255
        self.v_low = 122
        self.v_high = 234    
        self.trig = 0   

    def run(self,frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_blue = np.array([self.h_low,self.s_low,self.v_low])
        upper_blue = np.array([self.h_high,self.s_high,self.v_high])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)


        row_avg = np.sum(res,1)/frame.shape[1]

        row_avg = row_avg[:,0]

        n_max = np.amax(row_avg)

        if (n_max>=80):
            k = row_avg.argmax()
            k_val = np.amax(row_avg)
            row_avg[k-20:k+20] = 0
            t = row_avg.argmax()
            t_val = np.amax(row_avg)
            row_avg[t-20:t+20] = 0
            row_avg[k] = k_val
            row_avg[t] = t_val
            l = np.amax(np.where(row_avg>=80))
            cv2.line(frame,(0,l),(frame.shape[1],l),(255,255,255),4)

            return 1, l

        else:
            return 0,0




