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

import final_csv_final

Out_of_bounds = False
# LR_VAL = 1 #can be 0 : stay at place, 1 : move right, 2 : move left               #########################################################################################################################################################

# will have to go to left if 1st shelf khiski hui hogi toward left wrt to second

height = 180

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

        id = 1

        # global LR_VAL

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

        try:
            self.tello.move_left(40)
        except:
            pass
        try:
            self.tello.move_back(440)
        except:
            pass
            
        self.orient.orient(self.initial_yaw)									#uncomment this VERY IMPORTANT!!!!! 	

        goto_height(self.tello,height)											#the height of tello to reach 
        
        yaw = self.tello.get_yaw()

        self.warehouse.algo(yaw)

        self.orient.orient(yaw)

        self.warehouse.clear()

        # self.tello.move_left(50)													# to update

        print("have moved left")

        while(trig == 0):

            up, left, trig = self.rect_pass.run(yaw)
            print("up = {}, left = {}, trig = {}".format(up,left,trig))
            if(trig==0):
                self.after.run(left,up,yaw)
                self.after = after(self.tello)

        trig = 0

        # print("now going for after run")

        # self.after.run(2,0,yaw)


        # 3,0,yaw
        try:
        	self.tello.move_right(40)
        except:
        	pass

        # if(LR_VAL==1):
        #     self.tello.move_right(45)                #########################################################################################################################################################

        # elif(LR_VAL==2):
        #     self.tello.move_left(45)                 #########################################################################################################################################################

        self.after.run(left,up,yaw)
        self.after = after(self.tello)

        self.warehouse.algo(yaw)

        self.orient.orient(yaw)

        self.warehouse.clear()

        while(trig == 0):

            up, left, trig = self.rect_pass.run(yaw)

            if(trig==0):
                self.after.run(left,up,yaw)
                self.after = after(self.tello)

        # self.after.run(left,up,yaw)

        self.tello.move_forward(500)
        self.tello.move_forward(50)

        self.tello.land()
        self.tello.end()
        print("Ended")

        fname = "/home/carry/IMAV/IMAV_2019/NEW_FINAL_NO_TRACK/distribution.csv"

        # print("Enter input file name")
        # fname = input()
        # print("Enter ID")
        # id = input()

        final_csv_final.getout(id, fname)


def main():
    print("now i am gonna start the mission")
    tello = Tello()
    tello.offset = 60                             
    tello.connect()
    tello.streamoff()
    tello.streamon()
    start = hoohah(tello)
    start.run()
    # print("UP is {} and right is {}".format(a,b))

if __name__ == '__main__':
    main()


