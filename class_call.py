from djitellopy import Tello
import cv2
from class_FtextR_final import warehouse_R
from class_FtextL_final import warehouse_L

class warehouse_overall:
	def __init__(self, tello):
		self.tello = tello
		self.txt_R = warehouse_R(tello)
		self.txt_L = warehouse_L(tello)

	def algo(self):
		
		self.txt_L.scan()
		# GO_down_1.5m
		
		#self.txt_L.scan()
		# GO_up_1.5m
		