from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import re
from djitellopy import Tello
import imutils as imu
from time import sleep     
import os
import random
import time
import asyncio
import numpy as np
import cv2
import PIL.Image as pim
import pytesseract
from qrcode import *
from text import *
from imutils.object_detection import non_max_suppression



min_confidence = 0.5
height = width = 320
east = "opencv-text-recognition/frozen_east_text_detection.pb" 			#enter thefull path to east model
padding = 0.25


def roi_detect(image):
	orig = image.copy()
	(origH, origW) = image.shape[:2]

	# set the new width and height and then determine the ratio in change
	# for both the width and height
	(newW, newH) = (width, height)
	rW = origW / float(newW)
	rH = origH / float(newH)

	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet(east)

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# decode the predictions, then  apply non-maxima suppression to
	# suppress weak, overlapping bounding boxes
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# initialize the list of results
	results = []

	iter = 1

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
		print("text " + str(iter) + " :" + text)
		iter += 1



def decode_predictions(scores, geometry):
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



def apply_contrast(im):
	"""lab= cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

	#-----Splitting the LAB image to different channels-------------------------
	l, a, b = cv2.split(lab)

	#-----Applying CLAHE to L-channel-------------------------------------------
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	cl = clahe.apply(l)

	#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
	limg = cv2.merge((cl,a,b))

	#-----Converting image from LAB Color model to RGB model--------------------
	im = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)"""

	im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(im_hsv)
	ret, v = cv2.threshold(v,127,255,cv2.THRESH_BINARY)
	im_hsv[:, :, 2] = v
	im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
	return im

def apply_thresh(img):

	lower = np.array([50, 50, 50])
	upper = np.array([255, 255, 255])
	mask = cv2.inRange(img, lower, upper)
	img = cv2.bitwise_and(img, img, mask = mask)

tello = Tello()
tello.connect()
tello.streamoff()
tello.streamon()
rcout = np.zeros(4)

def img_resize(im):
	fx = 7.092159469231584126e+02
	fy = 7.102890453175559742e+02
	cx = 3.681653710406367850e+02
	cy = 2.497677007139825491e+02

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

# Main 
if __name__ == '__main__':


 
	f = open('warehouse.csv','w')
	f.write('%s,%s,\n'%("QR_Data", "Alphanum_text"))
	f.close()

	# Read feed
	#camera = cv2.VideoCapture(0)
	#capture = tello.get_video_capture()
	frame_read = tello.get_frame_read()

	while True:
		#ret, im = camera.read()
		
		frameBGR = np.copy(frame_read.frame)
		im = imu.resize(frameBGR, width=720)
		
		im, qrpoints, qrlist = main(im)
		if qrpoints != []:
	  		print(qrlist)

		#RESIZE
		#CROP

		#im = img_resize(im)
		#im = apply_contrast(im)
		#im = apply_thresh(im)



		#-------------------------------------------east part begins-------------------------------------

		roi_detect(im)


		#-----------------------------east part ends------------------------------



		print(len(qrlist))
		for i in range(len(qrlist)):
			Data = qrlist[i]
			"""
			#Print recognized text
			if text != "" and text != " ":
				text = text.replace('_', '')
				text = text.replace('\\', '')
				text = text.replace('/', '')
				text = text.replace('O', '0')
				print("text detected: %s"%(text))

				to_print = 0
				rex1 = re.compile("^[0-9]{2}[A-Z]$")
				rex2 = re.compile("^[0-9][A-Z]$")
				if rex1.match(text) or rex2.match(text):
					to_print = 1

				if(conf>60 and to_print):
					print()
					print()
					print("YAYAYAYAYAYYYY")
					print()
					print()
					f = open('warehouse.csv','a')
					f.write('%s,%s,\n'%(Data, text))
					f.close()
			"""
		cv2.imshow("Results", im)
		key = cv2.waitKey(1) & 0xFF;
		if key == ord("t"):
			tello.takeoff()    
		elif key == ord("l"):
			tello.land()
		elif key == ord("w"):
			rcOut[1] = 25
		elif key == ord("a"):
			rcOut[0] = -25
		elif key == ord("s"):
			rcOut[1] = -25
		elif key == ord("d"):
			rcOut[0] = 25
		elif key == ord("u"):
			rcOut[2] = 25
		elif key == ord("j"):
			rcOut[2] = -25
		elif key == ord("q"):
			breaks
		else:
			rcOut = [0,0,0,0]

		# print self.rcOut
		#tello.send_rc_control(int(rcOut[0]),int(rcOut[1]),int(rcOut[2]),int(rcOut[3]))
		#rcOut = [0,0,0,0]

	f.close()

tello.end()
#capture.release()
cv2.destroyAllWindows()
tello.streamoff()





	  
