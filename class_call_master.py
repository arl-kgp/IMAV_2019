from djitellopy import Tello
import cv2
from class_call import warehouse_overall
Out_of_bounds = False					# VARIABLE for detecting one end of shelf
tello = Tello()
tello.connect()
tello.streamoff()
tello.streamon()
txt = warehouse_overall(tello)
txt.algo(Out_of_bounds)
tello.streamoff()
tello.end()