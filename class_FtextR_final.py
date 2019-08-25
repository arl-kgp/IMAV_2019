from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import re
from djitellopy import Tello
import imutils as imu
from time import sleep  
import time   
import os
import random
import time
import asyncio
import numpy as np
import cv2
import pytesseract
from qrcode import *
from text import *
from imutils.object_detection import non_max_suppression
from PIL import Image
import scipy
import scipy.misc
from imutils.video import FPS



class warehouse_R:
	def __init__(self, tello):
		self.tello = tello
		self.rcout = np.zeros(4)
		self.east = "frozen_east_text_detection.pb" 			#enter the full path to east model
		print("[INFO] loading east text detector...")
		self.net = cv2.dnn.readNet(self.east)
		self.f = open('warehouse.csv','w')
		print("file opened")
		# cv2.waitKey(3000);
		self.hover_time = 0
		self.reached_qrcode = 0



	def text_better(self,text):
		list1 = list(text)
		if len(text) == 4:
			list1 = list1[1:]

		if(text[0]=='S'):
			list1[0]='5'
		if(text[0]=='I'):
			list1[0]='1'
		elif(text[0]=='A'):
			list1[0]='4'
		elif(text[0]=='O'):
			list1[0]='0'
		elif(text[0]=='Q'):
			list1[0]='0'


		if(text[1]=='S'):
			list1[1]='9'
		elif(text[1]=='I'):
			list1[1]='1'
		elif(text[1]=='A'):
			list1[1]='4'


		if(text[2]=='4'):
			list1[2]='A'
		elif(text[2]=='6'):
			list1[2]='C'
		elif(text[2]=='0'):
			list1[2]='D'
		elif(text[2]=='1'):
			list1[2]='I'
		elif(text[2]=='5'):
			list1[2]='S'
		elif(text[2]=='3'):
			list1[2]='B'
		elif(text[2]=='8'):
			list1[2]='B'

		text = ''.join(list1)
		return text

	def roi_detect(self,image):

		min_confidence = 0.5
		height = width = 320

		padding = 0.05

		orig = image.copy()
		# origH = 1080
		# origW = 720
		(origH, origW) = image.shape[:2]

		# set the new width and height and then determine the ratio in change
		# for both the width and height
		(newW, newH) = (width, height)
		rW = origW / float(newW)
		rH = origH / float(newH)

		# resize the image and grab the new image dimensions
		image = cv2.resize(image, (newW, newH))
		(H, W) = image.shape[:2]

		# define the two output layer names for the east detector model that
		# we are interested -- the first is the output probabilities and the
		# second can be used to derive the bounding box coordinates of text
		layerNames = [
			"feature_fusion/Conv_7/Sigmoid",
			"feature_fusion/concat_3"]

		# construct a blob from the image and then perform a forward pass of
		# the model to obtain the two output layer sets
		blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		self.net.setInput(blob)
		(scores, geometry) = self.net.forward(layerNames)

		# decode the predictions, then  apply non-maxima suppression to
		# suppress weak, overlapping bounding boxes
		(rects, confidences) = self.decode_predictions(scores, geometry, min_confidence)
		boxes = non_max_suppression(np.array(rects), probs=confidences)

		# initialize the list of results
		results = []

		iter = 1
		text_list = []
		conf_list = []
		corners = []

		output = orig.copy()

		for (startX, startY, endX, endY) in boxes:

			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)

			# in order to obtain a better OCR of the text we can potentially
			# apply a bit of padding surrounding the bounding box -- here we
			# are computing the deltas in both the x and y directions
			dX = int((endX - startX) * padding)
			dY = int((endY - startY) * padding)

			# apply padding to each side of the bounding box, respectively
			startX = max(0, startX - dX)
			startY = max(0, startY - dY)
			endX = min(origW, endX + (dX * 2))
			endY = min(origH, endY + (dY * 2))

			# extract the actual padded ROI
			roi = orig[startY:endY, startX:endX]
			
			im, text, conf = return_text(roi)

			cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
			corner_pts = [[startX, startY], [endX, endY]]
			
			print("text " + str(iter) + " :" + text)

			text = text.replace('_', '')
			text = text.replace('\\', '')
			text = text.replace('/', '')
			
			if len(text) == 3 or len(text) == 4:
				text = self.text_better(text)
				text_list.append(text)
				conf_list.append(conf)
				corners.append(corner_pts)
			iter += 1

		return text_list, conf_list, corners, output



	def decode_predictions(self,scores, geometry, min_confidence):
		# grab the number of rows and columns from the scores volume, then
		# initialize our set of bounding box rectangles and corresponding
		# confidence scores
		(numRows, numCols) = scores.shape[2:4]
		rects = []
		confidences = []

		# loop over the number of rows
		for y in range(0, numRows):
			# extract the scores (probabilities), followed by the
			# geometrical data used to derive potential bounding box
			# coordinates that surround text
			scoresData = scores[0, 0, y]
			xData0 = geometry[0, 0, y]
			xData1 = geometry[0, 1, y]
			xData2 = geometry[0, 2, y]
			xData3 = geometry[0, 3, y]
			anglesData = geometry[0, 4, y]

			# loop over the number of columns
			for x in range(0, numCols):
				# if our score does not have sufficient probability,
				# ignore it
				if scoresData[x] < min_confidence:
					continue

				# compute the offset factor as our resulting feature
				# maps will be 4x smaller than the input image
				(offsetX, offsetY) = (x * 4.0, y * 4.0)

				# extract the rotation angle for the prediction and
				# then compute the sin and cosine
				angle = anglesData[x]
				cos = np.cos(angle)
				sin = np.sin(angle)

				# use the geometry volume to derive the width and height
				# of the bounding box
				h = xData0[x] + xData2[x]
				w = xData1[x] + xData3[x]

				# compute both the starting and ending (x, y)-coordinates
				# for the text prediction bounding box
				endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
				endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
				startX = int(endX - w)
				startY = int(endY - h)

				# add the bounding box coordinates and probability score
				# to our respective lists
				rects.append((startX, startY, endX, endY))
				confidences.append(scoresData[x])

		# return a tuple of the bounding boxes and associated confidences
		return (rects, confidences)



	def apply_contrast(self,im):
		im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
		h,s,v = cv2.split(im_hsv)
		ret, v = cv2.threshold(v,127,255,cv2.THRESH_BINARY)
		im_hsv[:, :, 2] = v
		im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
		return im

	def apply_thresh(self,img):
		lower = np.array([50, 50, 50])
		upper = np.array([255, 255, 255])
		mask = cv2.inRange(img, lower, upper)
		img = cv2.bitwise_and(img, img, mask = mask)

	def img_resize(self,im):
		fx = 910.6412491
		fy = 680.16057188
		
		# cx = 3.681653710406367850e+02
		# cy = 2.497677007139825491e+02

		"""fx = 672.074266
		fy = 672.019640
		cx = 324.846853
		cy = 255.070573"""

		depth = 200
		real_text_w = 150	#200
		real_text_h = 60	#100
		favg = (fx+fy)/2
		text_w = (real_text_w*favg)/depth
		text_h = (real_text_h*favg)/depth

		optical_text_w = 172
		optimal_text_h = 74
		k = optimal_text_h/text_h
		rows = int(im.shape[0] * 1.2)
		cols = int(im.shape[1] * 1.2)
		dim = (cols, rows)
		resized = cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR)
		return resized


	def undistort(self,img):	
		
		balance = 1.0
		DIM=(960, 720)
		K=np.array([[676.4953507779437, 0.0, 485.52063689559924], [0.0, 673.0210851712478, 361.69019922623494], [0.0, 0.0, 1.0]])
		D=np.array([[0.17623414178050884], [-0.3648169258817548], [0.6005180717950186], [-0.3442054161739578]])
		dim1 = img.shape[:2][::-1]
		dim2 = dim1
		dim3 = dim1
		scaled_K = K * dim1[0] / DIM[0]
		scaled_K[2][2] = 1.0 
		new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
		map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
		undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

		return undistorted_img

	def hist_equalise(self,im):

		im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

		# equalize the histogram of the Y channel
		im_yuv[:,:,0] = cv2.equalizeHist(im_yuv[:,:,0])

		# convert the YUV image back to RGB format
		im = cv2.cvtColor(im_yuv, cv2.COLOR_YUV2BGR)
		return(im)

	def text_finder(self,im):

		# east + Tesseract
		text = None
		text_list, conf_list, corners, output = self.roi_detect(im)
		if(corners):
			print("Area: "+str(self.find_area(corners)))

		text_list_ref = []			## FINAL RETURN VALUES
		conf_list_ref = []
		corners_ref = []

		for index in range(len(text_list)):
			text = text_list[index]
			if self.check_format(text):
				text_list_ref.append(text_list[index])
				conf_list_ref.append(conf_list[index])
				corners_ref.append(corners[index])
				print("Added: "+str(text))
		if len(conf_list_ref) > 0:
			text = text_list_ref[np.argmax(conf_list)]
			corner_pts = corners_ref[np.argmax(conf_list)]
		
		return text, corners, output

	def check_format(self,text):
		to_print = False
		rex1 = re.compile("^[0-9]{2}[A-Z]$")
		rex2 = re.compile("^[0-9][A-Z]$")
		if rex1.match(text) or rex2.match(text):
			to_print = True

		return to_print

	def check_shelf_edge(self,src):
		gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
		gray = cv2.bitwise_not(gray)
		bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
		vertical = np.copy(bw)
		rows = vertical.shape[0]
		verticalsize = rows // 30

		# Create structure element for extracting vertical lines through morphology operations
		verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

		# Apply morphology operations
		vertical = cv2.erode(vertical, verticalStructure)
		vertical = cv2.dilate(vertical, verticalStructure)

		vertical = cv2.bitwise_not(vertical)

		# Step 1
		edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
		                            cv2.THRESH_BINARY, 3, -2)
		# Step 2
		kernel = np.ones((2, 2), np.uint8)
		edges = cv2.dilate(edges, kernel)

		# Step 3
		smooth = np.copy(vertical)
		# Step 4
		smooth = cv2.blur(smooth, (2, 2))
		# Step 5
		(rows, cols) = np.where(edges != 0)
		vertical[rows, cols] = smooth[rows, cols]
		return vertical

	def check_bars(self,img, src, barsize):
		rows, cols = img.shape
		min_avg_int = 255
		x = 0
		avg_int_list = []
		for i in range(cols-barsize):
			bar_img = img[:, i:i+barsize]
			avg_int = np.sum(bar_img)/(rows*barsize)
			avg_int_list.append(avg_int)
			
		avg_int_list = np.array(avg_int_list)

		idx = avg_int_list.argsort()[:15]
		coloured_list = []
		for x in idx:
			if np.sum(src[:,x]) != 0 and avg_int_list[x]<150:
				for i in range(10):
					if x+i<cols:
						src[:, x+i] = 0
		print(avg_int_list[idx])
		return src, idx

	def diff_shelf(self,im, qrpoints, textpoints):
		vert = self.check_shelf_edge(im)
		im, idx = self.check_bars(vert, im, 5)
		x1 = im.shape[1]
		for polygon in qrpoints:
			for point in polygon:
				if(x1>point.x):
					x1 = point.x
		print(textpoints)
		x2 = max(textpoints[0][0][0], textpoints[0][1][0])
		is_bar_present = 0
		print("x1: "+str(x1))
		print("x2: "+str(x2))
		
		for i in idx:
			if ((i-x1)*(i-x2)) < 0:
				is_bar_present += 1
		print("Is bar present "+str(is_bar_present))
		return is_bar_present
		
	def write_in_file(self,qrlist, text):

		for i in range(len(qrlist)):
			Data = qrlist[i]
			if(Data):
				Data = str(Data).strip("b'")
			# f = open('warehouse.csv','a')
			self.f.write('%s,%s,\n'%(Data, text))
			# f.close()

	def find_text_and_write(self,im, qrlist, qrpoints):
		text, corners, output = self.text_finder(im)
		print(text)
		
		check_text = 0                         # Flag to determine whether text actually found
		if text != None and corners:

			bars = self.diff_shelf(im, qrpoints, corners)
			if bars>8:
				return output, 2, corners

			check_text = 1
			self.write_in_file(qrlist, text)

		# check_text value:
		# 0->no text found
		# 1->text found, WRITE
		# 2->text found but different shelf

		return output, check_text, corners  

	def qr_intersection(self,lst1, lst2):
		ret_val = True 
		lst3 = [value for value in lst1 if value in lst2] 
		if len(lst3) > 0:
			ret_val = False
		return ret_val 

	def find_length(self,points):
		s = np.abs(points[0][0][0]-points[0][1][0])
		return s

	def find_area(self,points):
		A = np.abs(points[0][0][0]-points[0][1][0])*np.abs(points[0][0][1]-points[0][1][1])
		return A

	def rectifypos(self,frame):
		text_box_x_min = 3000
		text_box_x_max = 5000

		text, corners, output = self.text_finder_for_position(frame)
		if corners == []:
			return output, -1             # Too Far

		text_box_x = self.find_area(corners)
		print("AREA calculated: "+str(text_box_x))
		if text_box_x > text_box_x_max:   # More than Ideal
			fb = text_box_x_max-text_box_x
			return output, fb
		elif text_box_x < text_box_x_min: # Less than Ideal
			fb = text_box_x_min-text_box_x
			return output, fb 
		else:							  # Perfect
			return output, 0

	############### REDEFINED ROI_DETECT and TEXT_FINDER 
	def roi_detect_for_position(self,image):

		min_confidence = 0.5
		height = width = 320

		padding = 0.06

		orig = image.copy()
		# origH = 1080
		# origW = 720
		(origH, origW) = image.shape[:2]

		# set the new width and height and then determine the ratio in change
		# for both the width and height
		(newW, newH) = (width, height)
		rW = origW / float(newW)
		rH = origH / float(newH)

		# resize the image and grab the new image dimensions
		image = cv2.resize(image, (newW, newH))
		(H, W) = image.shape[:2]

		# define the two output layer names for the east detector model that
		# we are interested -- the first is the output probabilities and the
		# second can be used to derive the bounding box coordinates of text
		layerNames = [
			"feature_fusion/Conv_7/Sigmoid",
			"feature_fusion/concat_3"]

		# construct a blob from the image and then perform a forward pass of
		# the model to obtain the two output layer sets
		blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		self.net.setInput(blob)
		(scores, geometry) = self.net.forward(layerNames)

		# decode the predictions, then  apply non-maxima suppression to
		# suppress weak, overlapping bounding boxes
		(rects, confidences) = self.decode_predictions(scores, geometry, min_confidence)
		boxes = non_max_suppression(np.array(rects), probs=confidences)

		# initialize the list of results
		results = []

		iter = 1
		text_list = []
		conf_list = []
		corners = []

		output = orig.copy()

		for (startX, startY, endX, endY) in boxes:

			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)

			# in order to obtain a better OCR of the text we can potentially
			# apply a bit of padding surrounding the bounding box -- here we
			# are computing the deltas in both the x and y directions
			dX = int((endX - startX) * padding)
			dY = int((endY - startY) * padding)

			# apply padding to each side of the bounding box, respectively
			startX = max(0, startX - dX)
			startY = max(0, startY - dY)
			endX = min(origW, endX + (dX * 2))
			endY = min(origH, endY + (dY * 2))

			# extract the actual padded ROI
			roi = orig[startY:endY, startX:endX]
			
			im, text, conf = return_text(roi)

			cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
			corner_pts = [[startX, startY], [endX, endY]]
			
			#print("text " + str(iter) + " :" + text)

			text = text.replace('_', '')
			text = text.replace('\\', '')
			text = text.replace('/', '')
			
			text_list.append(text)
			conf_list.append(conf)
			corners.append(corner_pts)
			iter += 1

		return text_list, conf_list, corners, output

	def text_finder_for_position(self,im):

		# east + Tesseract
		text = None
		text_list, conf_list, corners, output = self.roi_detect_for_position(im)
		if len(conf_list) > 0:
			text = text_list[np.argmax(conf_list)]
			corner_pts = corners[np.argmax(conf_list)]
		
		return text, corners, output

	def motion_cmd_PID(self,size_diff):
		v_put = 0.005*size_diff
		return v_put

	def scan(self, dist):

		#cv2.namedWindow('Results',cv2.WINDOW_NORMAL)
		qrprev_list = []                                   # For comparing with newer qr-codes from next shelf
		qrlist = []
		check_qr_num = 0
		passed_shelf_var = False
		min_dist = -10
		max_dist = 20

		self.f.write('%s,%s,\n'%("QR_Data", "Alphanum_text"))
		# f.close()

		# Read feed:
		frame_read = self.tello.get_frame_read()
		self.rcout = [0,0,0,0]
		while True:
			                            
			# for FPS:
			start_time = time.time()

			# BATTERY checker
			self.tello.get_battery()

			frame = frame_read.frame

			# Undistortion --Uncomment for tello-001
			#frame = undistort(frame)

			# QR-codes detect
			k = cv2.waitKey(1) & 0xFF

			if k == ord("m"):
				
				if dist > max_dist:
					v = self.motion_cmd_PID(dist - max_dist)
					cv2.imshow("Results", output)
					self.tello.send_rc_control(0,int(v),0,0)
					print("Position correction: "+str(v))

					continue
				elif dist < min_dist:
					v = self.motion_cmd_PID(dist - min_dist)
					cv2.imshow("Results", output)
					self.tello.send_rc_control(0,int(v),0,0)
					print("Position correction:"+str(v))					
					continue

				print("m printing")
				frame, qrpoints, qrlist = main(frame)
				ret_val = self.qr_intersection(qrlist, qrprev_list)          #ret_val = 1 if no intersection otherwise 0

				print("intersection: "+str(ret_val))

				if qrpoints != [] and self.hover_time < 5 and ret_val == 1:	 # If QR detected, detect TEXT

					print(qrlist)
					self.rcout = [0,0,0,0]
					
					if self.reached_qrcode == 0:
						self.reached_qrcode=1
						print("New QRs found")

					frame, check_text, txt_corners = self.find_text_and_write(frame, qrlist, qrpoints)
					
					if check_text == 0:
						self.rcout = [5,0,0,0]
						print("text not found")
						self.tello.send_rc_control(int(self.rcout[0]),int(self.rcout[1]),int(self.rcout[2]),int(self.rcout[3]))
						cv2.imshow("Results",frame)
						continue

					if check_text == 2:
						# move until further code is detected.
						self.rcout = [5,0,0,0]
						print("text and QR in different Shelves")
						self.tello.send_rc_control(int(self.rcout[0]),int(self.rcout[1]),int(self.rcout[2]),int(self.rcout[3]))
						cv2.imshow("Results",frame)
						continue

					print("Text length: "+str(self.find_length(txt_corners)))
					self.hover_time = self.hover_time + time.time() - start_time 

				
				elif self.hover_time > 5:
					print("hover time: "+str(self.hover_time))

					self.rcout = [25,0,0,0]
					
					check_qr_num += 1
					
					self.hover_time = 0
					self.reached_qrcode = 0
					qrprev_list = qrlist
					
					print("exceeded time")
					if check_qr_num == 2:	# Since, ONLY 2 Shelves in experimental setup
						self.tello.land()
						break

				else:
					self.rcout = [10,0,0,0]
					self.reached_qrcode = 0
					self.hover_time= 0
		
			elif k == ord("t"):
				self.tello.takeoff()
			elif k == ord("l"):
				self.tello.land()
			elif k == ord("w"):
				# front
				self.rcout[1] = 50
			elif k == ord("a"):
				# left
				self.rcout[0] = -50
			elif k == ord("s"):
				# back
				self.rcout[1] = -50
			elif k == ord("d"):
				# right
				self.rcout[0] = 50
			elif k == ord("u"):
				# up
				self.rcout[2] = 50
			elif k == ord("j"):
				# down
				self.rcout[2] = -50
			elif k == ord("c"):
				self.rcout[3] = 50
			elif k == ord("v"):
				self.rcout[3] = -50
			elif k == ord("q"):
				self.f.close()
				self.tello.land()
				print("file closed")
				break

			cv2.imshow("Results",frame)
			self.tello.send_rc_control(int(self.rcout[0]),int(self.rcout[1]),int(self.rcout[2]),int(self.rcout[3]))
			self.rcout = [0,0,0,0]
			print("FPS: ", 1.0 / (time.time() - start_time))

		self.f.close()



		  
