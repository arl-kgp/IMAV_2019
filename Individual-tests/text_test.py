from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import re
import imutils as imu
from time import sleep  
import time   
import os
import random
import asyncio
import numpy as np
import cv2
import pytesseract
from text import *
from PIL import Image
import scipy
import scipy.misc
from imutils.object_detection import non_max_suppression

east = "/home/balaji/IMAV_2019/Individual-tests/imav-warehouse-management/opencv-text-recognition/frozen_east_text_detection.pb" 
net = cv2.dnn.readNet(east)

def roi_detect(image):

		min_confidence = 0.5
		height = width = 320

		padding = 0.06																				# to update padding

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
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)

		# decode the predictions, then  apply non-maxima suppression to
		# suppress weak, overlapping bounding boxes
		(rects, confidences) = decode_predictions(scores, geometry, min_confidence)
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

def decode_predictions(scores, geometry, min_confidence):
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

def text_finder(im):

	# east + Tesseract
	text = None
	text_list, conf_list, corners, output = roi_detect(im)
	# if(corners):
	# 	print("Area: "+str(self.find_area(corners)))

	text_list_ref = []			## FINAL RETURN VALUES
	conf_list_ref = []
	corners_ref = []

	# output = im.copy()

	for index in range(len(text_list)):
		text = text_list[index]
		if check_format(text):
			text_list_ref.append(text_list[index])
			conf_list_ref.append(conf_list[index])
			corners_ref.append(corners[index])
			print("Added: "+str(text))
	if len(conf_list_ref) > 0:   # Use in left
		text = text_list_ref[np.argmax(conf_list_ref)]
		corner_pts = corners_ref[np.argmax(conf_list_ref)]
	
	return text, corners, output

def check_format(text):
	to_print = False
	rex1 = re.compile("^[0-9]{2}[A-Z]$")
	rex2 = re.compile("^[0-9][A-Z]$")
	if rex1.match(text) or rex2.match(text):
		to_print = True

	return to_print

if __name__ == '__main__':

	cap = cv2.VideoCapture("/home/balaji/Videos/use.mp4")
	while(cap.isOpened()):
		print ("1")
		ret, frame = cap.read()
		output = frame
		text = []
		corners = []
		text, corners, output = text_finder(frame)
		print (text)
		cv2.imshow("frame",output)
		cv2.waitKey(1)








