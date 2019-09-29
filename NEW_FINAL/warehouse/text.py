from __future__ import print_function
import pytesseract
from pytesseract import Output
import numpy as np
import cv2

def return_text(im):

	# Define config parameters.
		# '-l eng'  for using the English language
		# '--oem 1' for using LSTM OCR Engine
	config = ('-l eng --oem 1 --psm 8 -c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM0123456789')
	#load_system_dawg=0 load_freq_dawg=0 --user-patterns /usr/share/tesseract-ocr/tessdata/eng.user-pattern 
	
	#config = ('-l eng --oem 1 --psm 3')
	config1 = ('--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

	#image_to_string
	#text = pytesseract.pytesseract.run_and_get_output(im, config="bazaaddr.config", extension = 'txt')
	#print("run_and_get_output: %s" %(text))

	#image_to_data
	d = pytesseract.image_to_data(im, config = config, output_type=Output.DICT)
	n_boxes = len(d['level'])
	index = 0
	max_conf = 0
	for i in range(n_boxes):
		(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
		cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
		box_text = d['text'][i]
		box_conf = int(d['conf'][i])
		if box_conf > max_conf and box_text != " ":
			max_conf = d['conf'][i]
			index = i
	text = d['text'][index]

	#text = pytesseract.image_to_string(im, config=config)
	
	return im, text, max_conf
