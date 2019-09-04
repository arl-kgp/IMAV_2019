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
		# tello.get_battery()
		# try:
		# 	self.tello.takeoff()
		# except:
		# 	print("takeoff to ho gya lol")
		# sleep(1)

		kp = 0.6
		ki = 0.01
		kd = 0.001

		e = target - self.tello.get_yaw()
		e_i = 0
		e_d = 0
		t1 = time()
		prev_e = e
		
		while(abs(e)): #for error
		# while True:
			# self.tello.get_battery()
			print(self.tello.get_yaw())
			e = target - self.tello.get_yaw()
			t2 = time()
			t = t2-t1
			print("t:" + str(t))
			e_i += e*t
			e_d = (e-prev_e)/t
			p = kp*e
			i = ki*e_i
			d = kd*e_d
			# print(p)
			# print(i)
			# print(d)
			S = p+i+d
			self.tello.send_rc_control(0,0,0,int(S))
			t1 = time()
			sleep(.01)
			prev_e = e
		# self.tello.land()
		# self.tello.end()

# tello = Tello()
# tello.connect()

# if __name__ == '__main__':
# 	Orient = Orient(tello)
# 	target = 5
# 	Orient.orient(target)