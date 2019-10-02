from djitellopy import Tello
import numpy as np

tello = Tello()
tello.connect()
tello.streamoff()
tello.streamon()

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
elif key == ord("c"):
	self.rcOut[3] = 50
elif key == ord("v"):
	self.rcOut[3] = -50


else:
    self.rcOut = [0,0,0,0]

# print self.rcOut
self.tello.send_rc_control(int(self.rcOut[0]),int(self.rcOut[1]),int(self.rcOut[2]),int(self.rcOut[3]))
self.rcOut = [0,0,0,0]

tello.end()
capture.release()
cv2.destroyAllWindows()

tello.streamoff()

