from djitellopy import Tello
import cv2
from warehouse.class_FtextR_final import warehouse_R
from warehouse.class_FtextL_final import warehouse_L
from warehouse.csvParserfinal import *
from warehouse.tello_height import *
from warehouse.orient_yaw import Orient as orient

class warehouse_overall:
	def __init__(self, tello):
		self.tello = tello
		self.txt_R = warehouse_R(tello)
		self.txt_L = warehouse_L(tello)
		self.orient = orient(self.tello)
		# self.height = self.tello.get_height()

	def algo(self,yaw):

		# goto_height(self.tello,100)												# to change height parameter

		# yaw = self.tello.get_yaw()

		self.orient.orient(yaw)
		
		self.txt_L.scan(yaw)
		h = self.tello.get_h()
		# # GO_down_1.5m

		# self.orient.orient(yaw)

		# k = h-150
		# if(k<20):																# change k minimum height
		k = 20

		goto_height(self.tello, k)

		self.orient.orient(yaw)
		
		self.txt_R.scan(yaw)

		self.orient.orient(yaw)
		# GO_up_1.5m

		parser1()

	def clear(self):
		self.txt_R = warehouse_R(self.tello)
		self.txt_L = warehouse_L(self.tello)



		