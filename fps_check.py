import numpy as np
from djitellopy import Tello
from imutils.video import FPS
import cv2
tello = Tello()
tello.connect()
tello.streamoff()
tello.streamon()
frame_read = tello.get_frame_read()
fps = None
fps = FPS().start()
while(True):
    cv2.imshow('frame',frame_read.frame)
    fps.update()
    fps.stop()
    print("\n FPS is : {} \n".format(fps.fps()))
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
tello.streamoff()
tello.end()
