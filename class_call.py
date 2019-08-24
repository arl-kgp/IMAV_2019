from djitellopy import Tello
import cv2
from class_FtextL_final import warehouse
tello =Tello()
txt = warehouse(tello)
cv2.waitKey(0)
txt.scan()