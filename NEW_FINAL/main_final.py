import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils as im

import sys
import os

import threading

from start_to_next.main import starting 
from warehouse.class_call import warehouse_overall 
from window_search.main import starting as rect_pass
from JoyStick_Controller.controller_module import Controller
import JoyStick_Controller.xbox as xbox

from tello_height import goto_height
from after_shelf import FrontEnd as after
from orient_yaw import Orient as orient

Out_of_bounds = False
LR_VAL = 0 #can be 0 : stay at place, 1 : move right, 2 : move left

# will have to go to left if 1st shelf khiski hui hogi toward left wrt to second

height = 185

class hoohah(object):

    def __init__(self,tello):

        self.tello = tello

        self.starting = starting(self.tello)

        self.initial_yaw = self.tello.get_yaw()

        self.warehouse = warehouse_overall(self.tello)

        self.rect_pass = rect_pass(self.tello)

        self.after = after(self.tello)

        self.orient = orient(self.tello)

    def run(self):

        global LR_VAL

        trig = 0

        try:
            self.tello.takeoff()
        except:
            print("Ab toh takeoff ho gya lol")
            
        
        time.sleep(2)


        print("Starting Controller")
        try:
            self.joy = xbox.Joystick()

            self.controller = Controller(self.tello, self.joy)
            x = threading.Thread(target=self.controller.run, daemon=True)
            x.start()
        except:
            print("Control fail ho gya lol")

        self.starting.run(self.initial_yaw)										#uncomment this VERY IMPORTANT!!!!! 	

        goto_height(self.tello,height)											#the height of tello to reach 
        
        yaw = self.tello.get_yaw()

        self.warehouse.algo(yaw)

        self.orient.orient(yaw)

        self.warehouse.clear()

        self.tello.move_left(50)													# to update

        print("have moved left")

        while(trig == 0):

            up, left, trig = self.rect_pass.run(yaw)
            print("up = {}, left = {}, trig = {}".format(up,left,trig))
            if(trig==0):
            	self.after.run(left,up,yaw)

        trig = 0

        # print("now going for after run")

        # self.after.run(2,0,yaw)


        # 3,0,yaw

        if(LR_VAL==1):
            self.tello.move_right(40)

        elif(LR_VAL==2):
            self.tello.move_left(40)

        self.after.run(left,up,yaw)

        self.warehouse.algo(yaw)

        self.orient.orient(yaw)

        self.warehouse.clear()

        while(trig == 0):

            up, left, trig = self.rect_pass.run(yaw)

        # self.after.run(left,up,yaw)

        self.tello.land()
        self.tello.end()
        print("Ended")


def main():
    print("now i am gonna start the mission")
    tello = Tello()
    tello.connect()
    tello.streamoff()
    tello.streamon()
    start = hoohah(tello)
    start.run()
    # print("UP is {} and right is {}".format(a,b))

if __name__ == '__main__':
    main()


