from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils as im
from start_to_next.toffNsrchShelf import FrontEnd as align_initial
from start_to_next.skip_shel import FrontEnd as skip_first
from orient_yaw import Orient as orient
from tello_height import goto_height
# import xbox

class starting(object):

    def __init__(self,tello):

        self.tello = tello
        # self.joy = xbox.Joystick()
        # self.tello.connect()
        # self.tello.streamoff()
        # self.tello.streamon()
        self.orient = orient(self.tello)
        self.align_initial = align_initial(self.tello)
        self.skip_first = skip_first(self.tello)
        self.align_next = align_initial(self.tello)

        self.height = 150                                                                         #to change along with line 40

    def run(self,init_yaw):

        # takeoff
        # try:
        #     self.tello.takeoff()
        # except:
        #     pass
        # time.sleep(2)

        self.orient.orient(init_yaw)
        # yaw orientation correct

        goto_height(self.tello,self.height)                                                      #to change along with line 26
        # go to height

        # read flag

        # rotate cloclwise/anti 180

        print("mission started")
        self.align_initial.run()                                                                 #todo variables
        print("Now have aligned in front of first frame")
        self.skip_first.run()
        self.skip_first.clear()
        print("skipped the first frame")
        self.align_next.run()
        print("Now have aligned in front of second frame")

        # self.skip_first.run()
        # self.skip_first.clear()

        # go more distance now
        self.tello.move_left(80)                                                                #declare how much to move right so that reach the shelf
        self.tello.rotate_clockwise(90)

        # do warehouse

        # search for the shelf to pass from

        # 


        # self.tello.land()
        # print("Ended")

def main():
    print("now i am gonna start the mission")
    tello = Tello()
    tello.connect()
    tello.streamoff()
    tello.streamon()
    start = starting(tello)
    start.run(0)

if __name__ == '__main__':
    main()


