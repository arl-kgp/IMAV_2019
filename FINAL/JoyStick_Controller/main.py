import xbox
from controller_module import FrontEnd
from djitellopy import Tello
from key_check import keyCheck

tello  =Tello()
tello.connect()
joy = xbox.Joystick()
keyC = keyCheck(tello,joy)
while 1:
	keyC.check()
