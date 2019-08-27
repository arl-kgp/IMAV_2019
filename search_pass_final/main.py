from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils as im


from left_right import FrontEnd as left_right
from up_down import FrontEnd as up_down
from passFrmWindow import FrontEnd as passWin
# from after_shelf import FrontEnd as after

class starting(object):

    def __init__(self):

        self.tello = Tello()
        # self.joy = xbox.Joystick()
        self.tello.connect()
        self.tello.streamoff()
        self.tello.streamon()

        self.left_right = left_right(self.tello)
        self.up_down = up_down(self.tello)
        self.passing = passWin(self.tello)
        # self.after = after(self.tello)

        self.up = 0
        self.right = 0
        self.trig = 0

    def run(self):

        while(1):

            self.trig, self.up = self.up_down.run()   #up down return two values
            self.up_down.clear()

            if(self.trig == 0):
                self.right = self.left_right.run(self.right)   #left right returns only one value
                self.left_right.clear()

            if(self.trig == 1):
                self.passing.run()
                # self.after.run(self.up,self.right)
                break

        self.tello.land()
        print("Ended")


def main():
    print("now i am gonna start the mission")
    start = starting()
    start.run()

if __name__ == '__main__':
    main()


