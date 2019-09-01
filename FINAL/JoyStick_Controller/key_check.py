import xbox
from controller_module import FrontEnd
from djitellopy import Tello

class keyCheck(object):

	def __init__(self,tello,joy):
		self.tello = tello
		self.joy = joy
		self.controller = FrontEnd(self.tello,self.joy)
	def check(self):
		if self.joy.leftTrigger() and self.joy.rightTrigger():
			self.controller.run()