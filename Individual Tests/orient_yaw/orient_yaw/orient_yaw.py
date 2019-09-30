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

		print("now in orientation wlalalalallalalalallalalalalalalalalallalalalalalallal")

		kp = 0.8
		ki = 0.012
		kd = 0.0008

		e = target - self.tello.get_yaw()
		e_i = 0
		e_d = 0
		t1 = time()
		prev_e = e
		
		while(abs(e)>2): #for error
		# while True:
			# self.tello.get_battery()
			print(self.tello.get_yaw())
			e = target - self.tello.get_yaw()
			sleep(.01)
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
			v = min(50,abs(int(S)))*(abs(int(S))/(int(S)))
			self.tello.send_rc_control(0,0,0,min(50,int(S)))
			t1 = time()
			
			prev_e = e
		self.tello.send_rc_control(0,0,0,0)
		print("Orientation wlae se nikla gya ab tohh sjbdygfvatsdusjnbsgvddbj")
		# self.tello.land()
		# self.tello.end()

# tello = Tello()
# tello.connect()

# if __name__ == '__main__':
# 	Orient = Orient(tello)
# 	target = 5
# 	Orient.orient(target)