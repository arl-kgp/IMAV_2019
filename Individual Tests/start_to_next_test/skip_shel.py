from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils as im

font = cv2.FONT_HERSHEY_COMPLEX

class FrontEnd(object):

    def __init__(self,tello):
        self.tello = tello
        # self.tello = Tello()

        # self.cap = cv2.VideoCapture(0)
        self.tracker = cv2.TrackerKCF_create()
        # self.tracker = cv2.CSRT_create()
        self.rcOut = np.zeros(4)
        self.bbox = (5,5,20,20)

        self.trigger = 0
        self.trigger_init = 0

        self.ar = 0

        self.ARmean = np.array([0])
        self.ARqueue = np.zeros((7,1))
        self.ARvar = np.array([0])

        self.lost = 0

        self.visible = 0

    def clear(self):

        self.tracker = cv2.TrackerKCF_create()
        # self.tracker = cv2.CSRT_create()
        self.rcOut = np.zeros(4)
        self.bbox = (5,5,20,20)

        self.trigger = 0
        self.trigger_init = 0

        self.ar = 0

        self.ARmean = np.array([0])
        self.ARqueue = np.zeros((7,1))
        self.ARvar = np.array([0])

        self.lost = 0

        self.visible = 0


    def run(self):
        
        # if not self.tello.connect():
        #     print("Tello not connected")
        #     return

        # # In case streaming is on. This happens when we quit this program without the escape key.
        # if not self.tello.streamoff():
        #     print("Could not stop video stream")
        #     return

        # if not self.tello.streamon():
        #     print("Could not start video stream")
        #     return 
        frame_read = self.tello.get_frame_read()

        should_stop = False

        while not should_stop:
            frame = frame_read.frame
            print(self.tello.get_bat())
            if frame_read.stopped:
                frame_read.stop()
                break
            cv2.imshow("original",frame)

            key = cv2.waitKey(1) & 0xFF;
            
            if (1):                                                               # changed automated key == ord("m")
                dst,mask = self.preproccessAndKey(frame)
                rect = self.get_coordinates(mask,dst)                                           # 0 3       order of rect coordinates
                if(self.trigger_init==0):                                                       # 1 2
                
                    if(rect[0][0] == 0):
                        continue
                    else:
                        print("hahahah")
                        self.trigger_init = 1

                if self.trigger_init == 1:

                    ok = self.start_tracking(rect,mask)
                    if(ok==True):
                        self.trigger_init = 2

                if self.trigger_init == 2:
            
                    self.track(mask)
                    if self.trigger == 1:
                        should_stop = True
                        # self.tello.land()
                        # self.tello.end()
                        break

            else:
                self.manualRcControl(key)

            self.sendRcControl()

    def manualRcControl(self,key):
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

    def sendRcControl(self):
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

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lowerbound = np.array([92,50,119])
        upperbound = np.array([117,255,255])

        mask = cv2.inRange(frame,lowerbound,upperbound)
        kernel2 = np.ones((3,3),np.uint8)#param 1
        mask = cv2.erode(mask,kernel2,iterations = 3)
        mask = cv2.dilate(mask,kernel2,iterations = 3)
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
            arSet = 0.8
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

        self.tracker = cv2.TrackerKCF_create()
        self.bbox = (rect[0][0],rect[0][1],rect[2][0]-rect[0][0],rect[2][1]-rect[0][1])
        ok = self.tracker.init(frame,self.bbox)
        return ok

    def track(self,frame):

        ok, self.bbox = self.tracker.update(frame)

        if ok:
            p1 = (int(self.bbox[0]), int(self.bbox[1]))
            p2 = (int(self.bbox[0]+ self.bbox[2]), int(self.bbox[1]+self.bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.imshow("with frame",frame)
            print("still visible")
            self.rcOut[0] = -30                                                  #to update speed along x axis
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
                self.visible = 0
                return
            print("LOST")
            self.visible = 0
            # self.lost +=1
            # if(self.lost>15):
            self.trigger = 1

def main():
    tello = Tello()
    frontend = FrontEnd(tello)

    frontend.run()

    print("found the end of the code")

if __name__ == '__main__':
    main()