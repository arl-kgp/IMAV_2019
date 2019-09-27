from djitellopy import Tello
import cv2
from class_FtextR_final import warehouse_R
from class_FtextL_final import warehouse_L
from csvParserfinal import *
from tello_height import *
from orient_yaw import Orient as orient

class warehouse_overall:
	def __init__(self, tello):
		self.tello = tello
		self.txt_R = warehouse_R(tello)
		self.txt_L = warehouse_L(tello)
		self.orient = orient(self.tello)

	def algo(self,yaw):

		goto_height(self.tello,170)												# to change height parameter

		# yaw = self.tello.get_yaw()
		
		self.txt_R.scan(yaw)
		h = self.tello.get_h()
		# GO_down_1.5m

		self.orient.orient(yaw)

		k = h-150
		if(k<30):																# change k minimum height
			k=30

		goto_height(self.tello, k)

		self.orient.orient(yaw)
		
		self.txt_L.scan(yaw)

		self.orient.orient(yaw)
		# GO_up_1.5m

		parser1()
		parserdash()
		parser2()

	def clear(self):
		self.txt_R = warehouse_R(self.tello)
		self.txt_L = warehouse_L(self.tello)



		