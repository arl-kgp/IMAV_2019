from djitellopy import Tello
import numpy as np
import cv2
import imutils as imu

def apply_contrast(im):
	im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(im_hsv)
	ret, v = cv2.threshold(v,127,255,cv2.THRESH_BINARY)
	im_hsv[:, :, 2] = v
	im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
	return im
def thresholdBGR(im):
	for i in range(im.shape[0]) :
		for j in range(im.shape[1]) :
			if ((im[i,j][0] < 15  and im[i,j][1] < 15 and im[i,j][2] < 15 ) or (im[i,j][0] >215  and im[i,j][1] > 215 and im[i,j][2] > 215 )) :
				im[i,j] = [255, 255, 255]
	return im



tello = Tello()
tello.connect()
tello.streamoff()
tello.streamon()

frame_read = tello.get_frame_read()

if __name__ == '__main__':
	while (True) :
		#ret, im = camera.read()
		frameBGR = np.copy(frame_read.frame)
		im = imu.resize(frameBGR, width=720)
		im = apply_contrast(im)
		im = thresholdBGR(im)
		cv2.imshow("v-thresh",im)
		cv2.waitKey(1)


tello.end()
#capture.release()
cv2.destroyAllWindows()
tello.streamoff()