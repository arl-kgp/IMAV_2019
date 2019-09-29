from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils as im
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

font = cv2.FONT_HERSHEY_COMPLEX

class FrontEnd(object):

    def __init__(self,tello):
        self.tello = tello
        # self.tello = Tello()

        # self.cap = cv2.VideoCapture(0)
        #self.tracker = cv2.TrackerKCF_create()
        self.tracker = cv2.TrackerKCF_create()
        self.rcOut = np.zeros(4)
        self.bbox = (5,5,20,20)

        self.trigger = 0
        self.prev_trigger = 0
        self.trigger_init = 0

        self.ar = 0

        self.ARmean = np.array([0])
        self.ARqueue = np.zeros((7,1))
        self.ARvar = np.array([0])

        self.lost = 0
        self.num_text_frames = 0
        self.visible = 0

    def detect_only_rectangle(self, frame):
        dst,mask = self.preproccessAndKey(frame)
        
        
        rect = self.get_coordinates(mask,dst)
        print("rect in func " + str(rect))
        if(rect[0][0] == 0):
            return False
        else:
            return True

    def run(self, frame):
       
        dst,mask = self.preproccessAndKey(frame)
        rect = self.get_coordinates(mask,dst)
        
        if(self.trigger_init==0):

            if(rect[0][0] == 0):
                return
            else:
                print("Triggered")
                self.trigger_init = 1

        if self.trigger_init == 1:
            # print(rect)
            # cv2.rectangle(dst, (rect[0][0], rect[0][1]), (rect[2][0], rect[2][1]),(0,0,255),3)
            # cv2.imshow("dst", dst)
            # time.sleep(2)
            ok = self.start_tracking(rect,mask)
            if(ok==True):
                self.trigger_init = 2

        if self.trigger_init == 2:

            self.track(mask)

        if self.prev_trigger == 0 and self.trigger == 1:
            self.num_text_frames += 1
        self.prev_trigger = self.trigger

    def run_updown(self, frame):
       
        dst,mask = self.preproccessAndKey(frame)
        rect = self.get_coordinates(mask,dst)
        
        if(self.trigger_init==0):
            # rect = self.get_coordinates(mask,dst)

            if(rect[0][0] == 0):
                return
            else:
                print("Triggered")
                self.trigger_init = 1

        if self.trigger_init == 1:
            # print(rect)
            # cv2.rectangle(dst, (rect[0][0], rect[0][1]), (rect[2][0], rect[2][1]),(0,0,255),3)
            # cv2.imshow("dst", dst)
            # time.sleep(2)
            ok = self.start_tracking(rect,mask)
            if(ok==True):
                self.trigger_init = 2

        if self.trigger_init == 2:

            self.track(mask)

        #if self.prev_trigger == 0 and self.trigger == 1:
        #    self.num_text_frames += 1
        self.prev_trigger = self.trigger
                

    def manualRcControl(self,key):		                        #commented manualRcControl
    	pass
        # if key == ord("w"):
        #     self.rcOut[1] = 50
        # elif key == ord("a"):
        #     self.rcOut[0] = -50
        # elif key == ord("s"):
        #     self.rcOut[1] = -50
        # elif key == ord("d"):
        #     self.rcOut[0] = 50
        # elif key == ord("u"):
        #     self.rcOut[2] = 50
        # elif key == ord("j"):
        #     self.rcOut[2] = -50
        # elif key == ord("l"):
        #     self.tello.land()
        # elif key == ord("t"):
        #     try:
        #         self.tello.takeoff()
        #     except:
        #         print("takeoff toh ho gya lol")
        #     time.sleep(2)
        # else:
        #     self.rcOut = [0,0,0,0]
        # return

    def sendRcControl(self):                                     # commented sendRcControl
    	# pass
        self.tello.send_rc_control(int(self.rcOut[0]),int(self.rcOut[1]),int(self.rcOut[2]),int(self.rcOut[3]))
        self.rcOut = [0,0,0,0]

        return

    def preproccessAndKey(self,frame_read):

        frameBGR = frame_read
        frame2use = im.resize(frameBGR,width=720)
            
        frame = frame2use 
        
        dst = frame2use            
        mask = self.getRectMask(dst)

        return dst,mask

    def getRectMask(self,frame):

        kernel = np.ones((5,5),np.uint8)#param 1

        frame = cv2.GaussianBlur(frame, (7, 7), 0)#param 1
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3,3), np.uint8)     

        frame_threshold = cv2.inRange(frame_HSV, (19, 0, 176), (64, 37, 255))
        frame_threshold = cv2.dilate(frame_threshold, kernel, iterations=5)
        frame_threshold = np.repeat(frame_threshold[:, :, np.newaxis], 3, 2)

        frame2 = np.uint8((frame_threshold//255)*np.int64(frame_HSV))

        frame_threshold = cv2.inRange(frame2, (20, 28, 73), (57, 139, 133))

        mask = frame_threshold
        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask


    def order_points(self, pts):

        pts = pts.reshape(4,2)
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
     
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        # print "dim",pts.shape
        # print "s",s 
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
     
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
     
        # return the ordered coordinates
        return rect


    def get_coordinates(self, mask,frame):

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = np.zeros((4,2), dtype ="float32")
        oldArea = 300
        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.012*cv2.arcLength(cnt, True), True) # 0.012 param
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            arSet = 0.5
            if area > 300:#param

                if len(approx) == 4:
                    if len(cnt) > 4:
                        (cx,cy),(MA,ma),angle = cv2.fitEllipse(cnt)
                        ar = MA/ma
                    else:
                        ar = (np.linalg.norm(approx[0] - approx[1]) + np.linalg.norm(approx[2] - approx[3]))/(np.linalg.norm(approx[2]-approx[1])+np.linalg.norm(approx[0]-approx[3]))
                        if ar > 1:
                            ar=1/ar

                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area)/hull_area

                    condition = ar < 1 and ar > arSet
                    if solidity > 0.9 and condition:

                        self.ar = ar
                        # print "ar",self.ar

                        self.ARqueue = np.roll(self.ARqueue,1,axis = 0)
                        self.ARqueue[0,:] = [ar]

                        self.ARvar = np.var(self.ARqueue,axis=0)
                        self.ARmean = np.mean(self.ARqueue,axis = 0)

                        if area > oldArea:
                            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                            cv2.circle(frame,(int(cx),int(cy)), 3, (0,0,255), -1)
                            cv2.putText(frame, "Rectangle" + str(angle), (x, y), font, 1, (0, 0, 0))

                            cntMain = approx
                            rect = self.order_points(cntMain)
                            # print("reached here")

                        oldArea = area

        return rect

    def start_tracking(self, rect,frame):

        print("tracking started")
        self.bbox = (rect[0][0],rect[0][1],rect[2][0]-rect[0][0],rect[2][1]-rect[0][1])
        ok = self.tracker.init(frame,self.bbox)
        return ok

    def track(self,frame):

        ok, self.bbox = self.tracker.update(frame)
        print("tracked\n\n\n")
        if ok:
            # p1 = (int(self.bbox[0]), int(self.bbox[1]))
            # p2 = (int(self.bbox[0]+ self.bbox[2]), int(self.bbox[1]+self.bbox[3]))
            # cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            # cv2.imshow("with frame",frame)
            print("still visible")
            self.rcOut[0] = 30
            self.rcOut[1] = 0
            self.rcOut[2] = 0
            self.rcOut[3] = 0
            self.trigger = 0
            self.lost = 0

            self.visible +=1

        else:
            if(self.visible<5):
                self.tracker = cv2.TrackerKCF_create()
                self.trigger_init = 1
                return
            print("LOST")
            self.visible = 0
            #self.lost +=1
            #if self.lost>5 :
            self.trigger = 1
            #self.tracker = cv2.TrackerKCF_create()
            self.tracker = cv2.TrackerKCF_create()
            self.tracker.clear()
            self.lost = 0
            self.trigger_init = 0

