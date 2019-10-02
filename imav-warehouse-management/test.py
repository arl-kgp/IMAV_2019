import cv2
import numpy as np
import re
camera = cv2.VideoCapture(0)
while False:
	ret, im = camera.read()
	cv2.imshow("win", im)
	key = cv2.waitKey(1) & 0xFF;
	if key == ord("a"):
		print("key found")
	else:
		pass
	#cv2.destroyAllWindows()
"""
img = cv2.imread("test.jpg")
mask = (img[:][:][0]<210)*(img[:][:][1]<210)*(img[:][:][2]<210)
mask = np.invert(mask)
img1 = np.multiply(mask, img)
cv2.imshow("labsdasjk",img1)
cv2.waitKey(0)
"""
"""
lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("lab",lab)

#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
cv2.imshow('l_channel', l)
cv2.imshow('a_channel', a)
cv2.imshow('b_channel', b)


#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)
cv2.waitKey(0)

#_____END_____#"""

text1 = "hsa"
text2 = "22A"
rex1 = re.compile("^[0-9]{2}[A-Z]$")
rex2 = re.compile("^[0-9][A-Z]$")
if rex1.match(text2) or rex2.match(text2):
	print("fsdfs")
else:
	print("NO")

texjs = "b'xsmn'"
rag = texjs.strip("b''")
print(rag)