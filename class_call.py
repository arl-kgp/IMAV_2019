from djitellopy import Tello
import cv2
from class_FtextR_final import warehouse_R
from class_FtextL_final import warehouse_L
from orbslam_tello_class import ORBSLAM
dist_from_shelf = 0                     # Distance from shelf (initially considered zero at 50cm)
orbslam = ORBSLAM()

class warehouse_overall:
	def __init__(self, tello):
		self.tello = tello
		self.txt_R = warehouse_R(tello)
		self.txt_L = warehouse_L(tello)

	def algo(self, Out_of_bounds):
		orbslam.run()
		print(orbslam.pose)
		"""
		while not Out_of_bounds:    	### EDIT, right bound
			self.txt_R.scan(dist)
		# GO_down_1.5m
		
		while not Out_of_bounds:		### EDIT, left bound
			self.txt_L.scan(dist)
		# GO_up_1.5m
		"""