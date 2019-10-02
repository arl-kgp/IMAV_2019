from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
import JoyStick_Controller.xbox

# Speed of the drone
S = 100
# Frames per second of the pygame window display
FPS = 25

class Controller():

    def __init__(self,tello,joy_passed):

        self.tello = tello
        self.joy = joy_passed
        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

    def run(self):

        while True:
            if self.joy.leftTrigger() and self.joy.rightTrigger():
                self.tello.manualControl = True
                
                leftStick = self.joy.leftStick()
                rightStick = self.joy.rightStick()
                self.keydown("leftst",leftStick)
                self.keydown("rightst",rightStick)

                if self.joy.Y():
                    self.keyup("takeoff")

                if self.joy.B():
                    self.keyup("land")

                if self.joy.X() and self.joy.leftBumper() and self.joy.rightBumper():			#killswitch
                    self.keyup("emergency")

                self.update()
            
            else:
                self.tello.manualControl = False

            #time.sleep(1 / FPS)

    def keydown(self, key, value):

        if key == "rightst":  # set roll/pitch velocity
            self.for_back_velocity = int(value[1]*S)         #pitch
            self.left_right_velocity = int(value[0]*S)       #roll
        elif key == "leftst": #set yaw/throttle
            self.yaw_velocity = int(value[0]*S)              #yaw
            self.up_down_velocity = int(value[1]*S)          #throttle

    def keyup(self, key):

        if key == "takeoff":  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == "land":  # land
            self.tello.land()
            self.send_rc_control = False
        elif key == "emergency":
        	self.tello.emergency()
        	self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                    self.yaw_velocity, 1)

    def printer(self):
        print("Yaw: "+str(self.yaw_velocity) + " Throttle: " + str(self.up_down_velocity) + " Pitch: " + str(self.for_back_velocity) + " Roll: " + str(self.left_right_velocity))