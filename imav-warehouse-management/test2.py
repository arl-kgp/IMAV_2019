from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils as im


from verticalScanNPass import FrontEnd as up_down
# from passFrmWindow import FrontEnd as passWin
# from after_shelf import FrontEnd as after
from orient_yaw import Orient as orient 

class starting(object):

    def __init__(self, tello):

        self.tello = tello
        # self.joy = xbox.Joystick()
        # print("nooooooooo")

        #self.left_right = left_right(self.tello)

        # print("yesssssssssssss")
        self.up_down = up_down(self.tello)
        # self.passing = passWin(self.tello)
        # self.after = after(self.tello)
        # print("outtttttttttttttt")

        self.orient = orient(self.tello)

        self.up = 0
        self.right = 0
        self.trig = 0

    def run(self):

        while(True):
            # print("yooooo")
            print("now in up down wali class")
            self.trig = self.up_down.run()
            self.up_down.clear()
          

        self.tello.land()
        print("Ended")
if __name__ == '__main__':
    pass

