from __future__ import print_function
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from text import *
import re


def near_shelf(im):
	
	retval = 0
	to_print = 0

	im, text, conf = return_text(im)
	if text != "" and text != " ":
		text = text.replace('_', '')
		text = text.replace('\\', '')
		text = text.replace('/', '')
		text = text.replace('O', '0')
		text = text.replace('F', '')
		
		#print(text)
		rex1 = re.compile("^[0-9]{2}$")
		rex2 = re.compile("^[0-9]$")
		if rex1.match(text) or rex2.match(text):
			to_print = 1
		if(conf>60 and to_print):
			shelf_code = text
			retval = 1
	return retval
	
camera = cv2.VideoCapture(0)
while True:
	ret, im = camera.read()
	if not ret:
		print("NOPES")
		break
	print(near_shelf(im))
	cv2.imshow("win", im)
	cv2.waitKey(1)
capture.release()
cv2.destroyAllWindows()
