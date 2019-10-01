from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils as im
from orient_yaw import Orient

tello = Tello()
tello.connect()
tello.streamoff()
tello.streamon()

print(tello.get_battery())
try:
	tello.takeoff()
except:
	print("unsuccessful")
orient = Orient(tello)
print ("initial yaw = "+str(tello.get_yaw()))
while 1:
	print("initial yaw = "+str(tello.get_yaw()))
	print ("Enter the yaw value:")

	yaw = int(input())
	if yaw == -1:
		break
	orient.orient(yaw)

	print ("Yaw rotation complete")
	print("final yaw = "+str(tello.get_yaw()))
	# time.sleep(2)

tello.land()