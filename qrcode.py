from __future__ import print_function

import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2


def PolyArea2D(pts):
	lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
	area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
	return area

def decode(im) : 
	# Find barcodes and QR codes
	decodedObjects = pyzbar.decode(im)

	# return results
	return decodedObjects

 
# Display barcode and QR code location  
def display(im, decodedObjects):

	# Loop over all decoded objects
	for decodedObject in decodedObjects: 
		points = decodedObject.polygon

		# If the points do not form a quad, find convex hull
		if len(points) > 4 : 
			hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
			hull = list(map(tuple, np.squeeze(hull)))
		else : 
			hull = points;
	 
		# Number of points in the convex hull
		n = len(hull)

		# Draw the convext hull
		for j in range(0,n):
			cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)

	# Return results
	return im

def main(im):
	decodedObjects = decode(im)

	qrpoints = []
	qrlist = []

	for obj in decodedObjects:
		area = PolyArea2D(obj.polygon)
		Type = obj.type
		if Type == "QRCODE":
			qrlist.append(obj.data)
			qrpoints.append(obj.polygon)
	
	im = display(im, decodedObjects)
	
	return im, qrpoints, qrlist
	
