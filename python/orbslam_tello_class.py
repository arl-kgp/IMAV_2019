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


class ORBSLAM(object):

    def __init__(self, tello,  camera_set):
        self.fc = 0
        self.cfc = 0
        self.buffer_frame = np.zeros((640, 720))
        self.cp = False
        self.tello = tello
        self.camera_set = camera_set
        self.pose = None

    def imageCallback(self):
        tello = self.tello
        capture = tello.get_video_capture()

        while 1:
            ret, frame = capture.read()
            frameBGR = np.copy(frame)
            if not self.cp:
                self.buffer_frame = im.resize(frameBGR,width=720)
            self.fc+=1

    def get_new_frame(self):
        while (self.fc == self.cfc or self.fc <= 5):
            sleep(0.01)
        self.cfc = self.fc
        self.cp = True
        rv = np.copy(self.buffer_frame)
        self.cp = False
        return rv

    def runslam(self):

        slam = orbslam2.System('ORBvoc.txt', self.camera_set, orbslam2.Sensor.MONOCULAR)
        slam.set_use_viewer(True)
        slam.initialize()


        while 1:
            image = self.get_new_frame()
            self.flag = 0
            pose = slam.process_image_mono(image, 0)
            if  not (pose is None):
                R = pose[:3,:3]
                T = pose[[0,1,2],[3]].reshape(3,1)
                self.pose = (-np.matmul(R.T, T))



    def run(self):
        p = Thread(target=self.imageCallback)
        p.start()

        p2 = Thread(target=self.runslam)
        p2.start()
        
        print('-----')
        print('Start processing sequence ...')
