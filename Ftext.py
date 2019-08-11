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
# import PIL.Image as pim
import pytesseract
from qrcode import *
from text import *
from imutils.object_detection import non_max_suppression
from PIL import Image
import scipy
import scipy.misc
from imutils.video import FPS


def text_better(text):
	list1 = list(text)
	if(len(text)==3) :
		if(text[0]=='S'):
			list1[0]='5'
		elif(text[0]=='I'):
			list1[0]='1'
		elif(text[0]=='A'):
			list1[0]='4'


		if(text[1]=='S'):
			list1[1]='5'
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

def roi_detect(image):

	min_confidence = 0.5
	height = width = 320

	padding = 0.03

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

		
		print("text " + str(iter) + " :" + text)

		text = text.replace('_', '')
		text = text.replace('\\', '')
		text = text.replace('/', '')
		
		if(len(text) == 3):
			text = text_better(text)
			text_list.append(text)
			conf_list.append(conf)
		iter += 1

	return text_list, conf_list, output



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



def apply_contrast(im):
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
east = "frozen_east_text_detection.pb" 			#enter the full path to east model
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(east)
f = open('warehouse.csv','w')
print("file opened")
# cv2.waitKey(3000);

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


def undistort(im):
	K = np.array([[5.277994366999020031e+02,0.000000000000000000e+00,3.711893449350251331e+02], [0.000000000000000000e+00,5.249025134499009937e+02,2.671209192674019732e+02], [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00
	]], dtype = 'uint8')

	dist = np.array([-1.941494206892808161e-01,-1.887639714668869206e-02,-5.988986169837847741e-03,-7.372353351255917582e-05,7.269696522356267065e-02], dtype = 'uint8')

	K_inv = np.linalg.inv(K)

	h , w = im.shape[:2]

	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

	mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
	dst = cv2.remap(im,mapx,mapy,cv2.INTER_LINEAR)

	x,y,w,h = roi
	im = dst[y:y+h,x:x+w]

	#print("ROI: ",x,y,w,h)
	#cv2.imshow("lkgs",frame2use)

	return(im)

def hist_equalise(im):

	im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	im_yuv[:,:,0] = cv2.equalizeHist(im_yuv[:,:,0])

	# convert the YUV image back to RGB format
	im = cv2.cvtColor(im_yuv, cv2.COLOR_YUV2BGR)
	return(im)

def text_finder(im):

	# EAST + Tesseract
	text = None
	text_list, conf_list, output = roi_detect(im)
	if len(conf_list) > 0:
		text = text_list[np.argmax(conf_list)]

	return text, output

def check_format(text):
	to_print = 0
	rex1 = re.compile("^[0-9]{2}[A-Z]$")
	rex2 = re.compile("^[0-9][A-Z]$")
	if rex1.match(text) or rex2.match(text):
		to_print = 1

	return to_print

def write_in_file(qrlist, text):

	for i in range(len(qrlist)):
		Data = qrlist[i]
		# f = open('warehouse.csv','a')
		f.write('%s,%s,\n'%(Data, text))
		# f.close()

def find_text_and_write(im, qrlist):
	text, output = text_finder(im)
	print(text)
	check = 1
	#check = check_format(text)

	if text != None and check:
		write_in_file(qrlist, text)

	return output

		
if __name__ == '__main__':


	f.write('%s,%s,\n'%("QR_Data", "Alphanum_text"))
	# f.close()

	# Read feed:
	frame_read = tello.get_frame_read()

	while True:
		
		# for FPS:
		start_time = time.time()
		# fps = FPS().start()
		# Frame preprocess
		# frameBGR = np.copy(frame_read.frame)
		im = imu.resize(frame_read.frame, width=720)
		# b,g,r = cv2.split(im)
		# kernel = np.ones((5,5), np.uint8)
		# img_erosion = cv2.erode(b, kernel, iterations=1)
		# img_dilation = cv2.dilate(b, kernel, iterations=1)
		# img_erosion = cv2.erode(g, kernel, iterations=1)
		# img_dilation = cv2.dilate(g, kernel, iterations=1)
		# img_erosion = cv2.erode(r, kernel, iterations=1)
		# img_dilation = cv2.dilate(r, kernel, iterations=1)
		# im[:,:,0]=b
		# im[:,:,1]=g
		# im[:,:,2]=r
		# Undistortion --Uncomment for Tello-001
		#im = undistort(im)

		# QR-codes detect
		im, qrpoints, qrlist = main(im)

		# if qrpoints != []:			# If QR detected, detect TEXT
		print(qrlist)

		#im = img_resize(im)
		#im = apply_contrast(im)
		#im = apply_thresh(im)
		#im = hist_equalise(im)

		im = find_text_and_write(im, qrlist)
				

		cv2.imshow("Results", im)
		key = cv2.waitKey(1) & 0xFF;
		if key == ord("t"):
			tello.takeoff()    
		elif key == ord("l"):
			tello.land()
		elif key == ord("w"):
			rcOut[1] = 10
		elif key == ord("a"):
			rcOut[0] = 10
		elif key == ord("s"):
			rcOut[1] = -10
		elif key == ord("d"):
			rcOut[0] = -10
		elif key == ord("u"):
			rcOut[2] = 10
		elif key == ord("j"):
			rcOut[2] = -10
		elif key == ord("c"):
			rcOut[3] = 10
		elif key == ord("v"):
			rcOut[3] = -10
		elif key == ord("q"):
			f.close()
			print("file closed")
			break
		else:
			rcOut = [0,0,0,0]

		#print self.rcOut
		tello.send_rc_control(int(rcOut[0]),int(rcOut[1]),int(rcOut[2]),int(rcOut[3]))
		rcOut = [0,0,0,0]
		# fps.update()
		# fps.stop()
		# print("FPS: ", fps.elapsed())
		print("FPS: ", 1.0 / (time.time() - start_time))



tello.end()
cv2.destroyAllWindows()
tello.streamoff()





	  
