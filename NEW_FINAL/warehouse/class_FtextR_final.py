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
import asyncio
import numpy as np
import cv2
import pytesseract
from qrcode import *
from text import *
from imutils.object_detection import non_max_suppression
from shelf_trav import FrontEnd as shelf_traversal
from PIL import Image
import scipy
import scipy.misc
from imutils.video import FPS
from align_to_frame import FrontEnd as correct_position # Use in left

from align_rect import FrontEnd as align_rect

class warehouse_R:
	def __init__(self, tello):
		self.tello = tello
		self.rcout = np.zeros(4)
		self.east = "frozen_east_text_detection.pb" 			#enter the full path to east model
		print("[INFO] loading east text detector...")
		self.net = cv2.dnn.readNet(self.east)
		self.f = open('warehouse.csv','w')
		self.f1 = open('out2.csv','w')
		print("file opened")
		# cv2.waitKey(3000);
		self.hover_time = 0
		self.reached_qrcode = 0
		self.should_stop = False
		self.align = correct_position(tello)  # Use in left

		self.align_rect = align_rect(self.tello)

		self.yaw = self.tello.get_yaw()


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
		if len(conf_list_ref) > 0:   # Use in left
			text = text_list_ref[np.argmax(conf_list_ref)]
			corner_pts = corners_ref[np.argmax(conf_list_ref)]
		
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
		edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
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
		y = 0
		avg_int_list = []
		idx_new = []
		for i in range(cols-barsize):
			bar_img = img[:, i:i+barsize]
			avg_int = np.sum(bar_img)/(rows*barsize)
			avg_int_list.append(avg_int)
			
		avg_int_list = np.array(avg_int_list)

		idx = avg_int_list.argsort()[:15]														# minimum 15 to update
		coloured_list = []
		for x in idx:
			if avg_int_list[x]<150:
				idx_new[y] = x
				y = y+1
			if np.sum(src[:,x]) != 0 and avg_int_list[x]<150:									# to update
				for i in range(10):
					if x+i<cols:
						src[:, x+i] = 0
		print(avg_int_list[idx])
		cv2.imshow("bars", src)
		cv2.waitKey(0)
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
			if bars>15:
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

		frame_p, qrpoints, qrlist = main(frame)
		if qrpoints != []:
			frame = frame_p

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

	# GO UP , include in leftleft
	def go_up(self, frame, qrprev_list):
		
		frame, qrpoints, qrlist = main(frame)
		
		#dst,mask = frontend.preproccessAndKey(frame)
		#rect = frontend.get_coordinates(mask,dst)
		#if rect[0][0] == 0:
		#	return False
		self.write_in_file2(qrprev_list, qrlist)
		return True

	def write_in_file2(self, prev_qr, qrlist):
		for j in range(len(prev_qr)):
			prev_Data = prev_qr[j]
			if prev_Data:
				prev_Data = str(prev_Data).strip("b'")
				for i in range(len(qrlist)):
					Data = str(qrlist[i]).strip("b'")
					if(Data):
						Data = str(Data).strip("b'")
					self.f1.write('%s,%s,\n'%(prev_Data, Data))
	

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

	def correct_pos(self, frame):
		output, feedback = self.rectifypos(frame)
		print("feedback: "+str(feedback))
		dist = self.motion_cmd_PID(feedback)
		if dist>0:
			dist = dist*2
		if feedback == -1:
			rcOut = [0,10,0,0]
			print("Not visible, forward")
		elif feedback == 0:
			rcOut = [0,0,0,0]
			print("No deviation")
		else:
			rcOut = [0,dist,0,0]
			print("distance: "+str(dist))
			print("PID cmd")
		return rcOut, output

	def scan(self,yaw):

		self.yaw = yaw

		total_time = 0
		#cv2.namedWindow('Results',cv2.WINDOW_NORMAL)
		qrprev_list = []                                   # For comparing with newer qr-codes from next shelf
		qrlist = []
		check_qr_num = 0
		passed_shelf_var = False
		should_correct_pos = False
		num_corrections = 0
		go_up = False
		go_down = False
		start_height = 0
		should_track = False
		vertical_motion = False
		
		align_without_QR = False
		rectangle_without_QR = False

		self.f.write('%s,%s,\n'%("QR_Data", "Alphanum_text"))
		# f.close()

		# Read feed:
		frame_read = self.tello.get_frame_read()

		## Shelf Traversal Class Object
		trav1 = shelf_traversal(self.tello)
		self.rcout = [0,0,0,0]
		while True:
										
			# for FPS:
			start_time = time.time()

			# BATTERY checker
			try:  # Use in left
				self.tello.get_battery()
			except:
				pass

			frame = frame_read.frame

			# Undistortion --Uncomment for tello-001
			#frame = undistort(frame)

			# QR-codes detect
			k = cv2.waitKey(1) & 0xFF

			if 1:                                                  ###k == ord("m")   updated automated
				
				if go_up or go_down:
					vertical_motion = True
				else:
					vertical_motion = False

				##################################### NOT SURE ON THIS tho, Loses tracking perhaps
				#if not vertical_motion:
				#print("No vertical motion, tracking")
				
				initial_no_of_frames = trav1.num_text_frames
				if vertical_motion:
					trav1.run_updown(frame)  # Use in left
				else:
					trav1.run(frame)  # Use in left
				cv2.destroyWindow("dst")
				print("text_frames_detected: " + str(trav1.num_text_frames))

				if ((trav1.num_text_frames - initial_no_of_frames) > 0) and align_without_QR:
					align_without_QR = False

				if trav1.num_text_frames == 4:              # NO. of shelves in one row # 4
					self.should_stop = True
					print("Finished")
					break

				rectangle_without_QR = trav1.detect_only_rectangle(frame)
				print("rectangle = "+str(rectangle_without_QR))

				## TEXT box detection and position correction
				"""
				if should_correct_pos == True:
					print("Correcting.....")
					rcOut, output = self.correct_pos(frame)
					cv2.imshow("Results", output)
					self.tello.send_rc_control(int(rcOut[0]),int(rcOut[1]),int(rcOut[2]),int(rcOut[3]))
					# BREAK STATEMENT
					if int(rcOut[1]) == 0:
						num_corrections += 1		# Increase num_corrections by one for "NO Deflection"
					if num_corrections == 3:		# Break when 3 correct predictions
						num_corrections = 0			# num_corrections reset
						should_correct_pos = False	
					continue
				"""

				# Use in left -->
				## Text Box detection from coloured frames
				if should_correct_pos == True:

					self.align_rect.run(self.yaw)
					print("exited align_rect")
					self.align_rect.clear()
					should_correct_pos = False
					#NOT SURE
					trav1.prev_trigger = 1
					trav1.trigger = 1
					
					continue

				# leftleft
				if go_up:
					print("up")
					present_height = self.tello.get_h()
					if (present_height - start_height) > 75:
						go_up = False
						go_down = True
						cv2.imshow("Results",frame)
						continue
					check = self.go_up(frame, qrprev_list)
					if (not check or trav1.detect_only_rectangle(frame)) and (present_height-start_height)>40:
						go_up = False
						go_down = True
						cv2.imshow("Results",frame)
						continue
					#self.rcOut = [0, 0, 10, 0]
					self.tello.send_rc_control(0, 0, 20, 0)
					cv2.imshow("Results",frame)
					continue

				if go_down:
					print("down")
					#if self.tello.get_h() <= start_height and trav1.detect_only_rectangle(frame):
					if self.tello.get_h() <= start_height:
						go_down = False
						#trav1.num_text_frames = trav1.num_text_frames-1
						should_correct_pos = True
						continue

					#self.rcOut = [0, 0, -10, 0]
					self.tello.send_rc_control(0, 0, -20, 0)
					cv2.imshow("Results",frame)
					continue

				print("m printing")
				frame_1, qrpoints, qrlist = main(frame)
				ret_val = self.qr_intersection(qrlist, qrprev_list)          #ret_val = 1 if no intersection otherwise 0

				print("intersection: "+str(ret_val))

				if qrpoints != [] and self.hover_time < 3 and ret_val == 1:	 # If QR detected, detect TEXT

					frame = frame_1
					print(qrlist)
					self.rcout = [0,0,0,0]
					
					if self.reached_qrcode == 0:
						self.reached_qrcode=1
						print("New QRs found")

					frame, check_text, txt_corners = self.find_text_and_write(frame, qrlist, qrpoints)
					
					if check_text == 0:
						# if text is not visible and qr is visible 
						self.rcout = [5,0,0,0]
						print("text not found")
						self.tello.send_rc_control(int(self.rcout[0]),int(self.rcout[1]),int(self.rcout[2]),int(self.rcout[3]))
						cv2.imshow("Results",frame)
						continue

					if check_text == 2:
						# move until further code is detected. 
						# pichle shelf ka alphanumeric and current ka qr
						self.rcout = [5,0,0,0]
						print("text and QR in different Shelves")
						self.tello.send_rc_control(int(self.rcout[0]),int(self.rcout[1]),int(self.rcout[2]),int(self.rcout[3]))
						cv2.imshow("Results",frame)
						continue

					print("Text length: "+str(self.find_length(txt_corners)))
					self.hover_time = self.hover_time + time.time() - start_time 

				
				elif self.hover_time > 3:
					print("hover time: "+str(self.hover_time))

					self.rcout = [15,0,0,0]
					
					check_qr_num += 1
					
					self.hover_time = 0
					self.reached_qrcode = 0
					qrprev_list = qrlist
					
					print("exceeded time")
					#if check_qr_num == 2:	# Since, ONLY 2 Shelves in experimental setup
					#	self.tello.land()
					#	break

					# leftleft
					start_height = self.tello.get_h()
					go_up = True
					

					should_correct_pos = True  # Use in left

				elif rectangle_without_QR and not align_without_QR:

					should_correct_pos = True  # Use in left
					align_without_QR = True
					print("No QR, ONLY text")

				else:
					total_time = total_time + time.time() - start_time
					#if total_time > 5:              
					#	should_correct_pos = True   			   ## Just ONCE every time this distance is reached
					#	total_time = 0
					self.rcout = [15,0,0,0]
					self.reached_qrcode = 0
					self.hover_time= 0
		
			# elif k == ord("t"):  	                               ### removed manual control
			# 	try:
			# 		self.tello.takeoff()
			# 	except:
			# 		print("takeoff done")
			# 	time.sleep(2)

			# elif k == ord("l"):
			# 	self.tello.land()
			# elif k == ord("w"):
			# 	# front
			# 	self.rcout[1] = 50
			# elif k == ord("a"):
			# 	# left
			# 	self.rcout[0] = -50
			# elif k == ord("s"):
			# 	# back
			# 	self.rcout[1] = -50
			# elif k == ord("d"):
			# 	# right
			# 	self.rcout[0] = 50
			# elif k == ord("u"):
			# 	# up
			# 	self.rcout[2] = 50
			# elif k == ord("j"):
			# 	# down
			# 	self.rcout[2] = -50
			# elif k == ord("c"):
			# 	self.rcout[3] = 50
			# elif k == ord("v"):
			# 	self.rcout[3] = -50
			# elif k == ord("q"):
			# 	self.f.close()
			# 	self.f1.close()
			# 	self.tello.land()
			# 	print("file closed")
			# 	break

			cv2.imshow("Results",frame)
			self.tello.send_rc_control(int(self.rcout[0]),int(self.rcout[1]),int(self.rcout[2]),int(self.rcout[3]))
			self.rcout = [0,0,0,0]
			print("FPS: ", 1.0 / (time.time() - start_time))

		self.f.close()
		self.f1.close()




		  
