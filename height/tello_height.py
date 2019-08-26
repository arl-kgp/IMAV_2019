#!/usr/bin/env python3

from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
import imutils as im
import math
import matplotlib.pyplot as plt

import sys
import os.path
import orbslam2
import time
import cv2
import numpy as np
import asyncio
from time import sleep

from threading import Thread
font = cv2.FONT_HERSHEY_SIMPLEX


class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()
        self.rcOut=np.zeros(4)
        
       
        self.flag = 0
        self.frame = None

        self.Height = 200
        # self.telloPose = np.array([])
            # self.telloEulerAngles = EulerA



    def run(self):
        if not self.tello.connect():
            print("Tello not connected")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return
        should_stop = False
        land = False
        start_h = False
        off = False
        while not should_stop:

            cv2.imshow('win',np.zeros((800,800)))
            key = cv2.waitKey(1) & 0xFF

            # gray = cv2.cvtColor(frame2use, cv2.COLOR_BGR2GRAY)
            self.flag = 0
            
            #self.FindPose(result,K_inv)
            self.rcOut = [0,0,0,0]
            h = self.tello.get_h()
            if not land:
                if key == ord("w"):
                    self.rcOut[1] = 50
                elif key == ord("a"):
                    self.rcOut[0] = -50
                elif key == ord("s"):
                    self.rcOut[1] = -50
                elif key == ord("d"):
                    self.rcOut[0] = 50
                elif key == ord("u"):
                    self.rcOut[2] = 50
                elif key == ord("j"):
                    self.rcOut[2] = -50
                else:
                    self.rcOut = [0,0,0,0]
                
                if key == ord("p"):
                    start_h = True
                
                if start_h:
                    if abs(self.Height- h) > 40 :
                        self.rcOut[2] = int((self.Height - h) * 0.3)
                    elif abs(self.Height- h) > 15:
                        self.rcOut[2] = int((self.Height - h) * 0.5)
                    elif abs(self.Height- h) <= 15 :
                        if self.Height == 200:
                            print('300 now!!!!!!!!!!!!!!')
                            self.Height = 300
                        elif self.Height == 300:
                            print('100 now!!!!!!!!!!!!!!')
                            self.Height = 100
                        elif self.Height == 100:
                            print('200 now!!!!!!!!!!!!!!')
                            self.Height = 200



            print(h-self.Height)
            try:
                self.tello.send_rc_control(int(self.rcOut[0]),int(self.rcOut[1]),int(self.rcOut[2]),int(self.rcOut[3]))

                if key == ord("q"):
                    break
                if key == ord("t") and not off:
                    self.tello.takeoff()
                    off = True    
                    print('took off')
                if key == ord("l"):
                    self.tello.land()
                    land = True
                    start_h = False
            except:
                pass
            self.rcOut = [0,0,0,0]

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sleep(1/60.0)
        # Call it always before finishing. I deallocate resources.
        self.tello.end()


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()


            