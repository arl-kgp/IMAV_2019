from djitellopy import Tello
import cv2
from class_call import warehouse_overall
Out_of_bounds = False					# VARIABLE for detecting one end of shelf
tello = Tello()
tello.connect()
tello.streamoff()
tello.streamon()
txt = warehouse_overall(tello)
yaw = tello.get_yaw()
# try:
# 	tello.takeoff()
# except:
# 	pass
txt.algo(yaw)
tello.land()
tello.streamoff()
tello.end()