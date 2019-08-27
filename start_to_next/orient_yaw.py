from djitellopy import Tello
from time import sleep

class Orient(object):

	def __init__(self,tello):
		self.tello = tello
		self.tello.set_speed(10)

	def orient(self,target):
		print(tello.get_bat())
		try:
			self.tello.takeoff()
		except:
			print("takeoff to ho gya lol")
		sleep(3)

		while(abs(self.tello.get_yaw()-target)>5): #for error
		# while True:
			print(self.tello.get_yaw())
			sleep(0.01)
			S = 0.4*(abs(self.tello.get_yaw()-target)+5)
			if((self.tello.get_yaw()-target)>0):
				self.tello.send_rc_control(0,0,0,-int(S))

			else:
				self.tello.send_rc_control(0,0,0,int(S))


if __name__ == '__main__':
	tello = Tello()
	tello.connect()
# S=10
	Orient = Orient(tello)
	target = 5
	Orient.orient(target)