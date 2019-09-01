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

from time import sleep


def goto_height(tello, Height):
  

    should_stop = False
    land = False
    start_h = False
    off = False
    while not should_stop:


        # gray = cv2.cvtColor(frame2use, cv2.COLOR_BGR2GRAY)
        
        #FindPose(result,K_inv)
        rcOut = [0,0,0,0]
        h = tello.get_h()
     
        if abs(Height- h) > 40 :
            rcOut[2] = int((Height - h) * 0.3)
        elif abs(Height- h) > 15:
            rcOut[2] = int((Height - h) * 0.5)
        elif abs(Height- h) <= 15 :
            return


        print(h-Height)
        try:
            tello.send_rc_control(int(rcOut[0]),int(rcOut[1]),int(rcOut[2]),int(rcOut[3]))
        except:
            pass
        rcOut = [0,0,0,0]


        sleep(1/60.0)
    # Call it always before finishing. I deallocate resources.