import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from djitellopy import Tello
from time import sleep,time

class Orient(object):

	def __init__(self,tello):
		self.tello = tello
		self.tello.set_speed(10)

	def orient(self,target):
			
			e = target - self.tello.get_yaw()
			if(e>0):
				e+=2
			else:
				e-=2
			if(e>0):
				self.tello.rotate_clockwise(int(e))
			else:
				self.tello.rotate_counter_clockwise(-int(e))

			# sleep(1.7)