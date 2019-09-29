# go down until you find the rectangle, after finfing it, align to it and then use the skip shelve functions and all.

from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils as im

from align_rect import FrontEnd as align_rect

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
        self.trigger_init_start = 0

        self.trigger_init = 0

        self.ar = 0

        self.ARmean = np.array([0])
        self.ARqueue = np.zeros((7,1))
        self.ARvar = np.array([0])

        self.lost = 0

        self.visible = 0

        self.align_rect = align_rect(self.tello)

    def run(self,left,up,yaw):

        frame_read = self.tello.get_frame_read()

        should_stop = False

        # go down and search, align
        # if up == 1 go left
        # if up == 0 go up and then go left

        print("now here")

        while not should_stop:
            frame = frame_read.frame
            print(self.tello.get_bat())
            if frame_read.stopped:
                frame_read.stop()
                break

            cv2.imshow("original",frame)

            key = cv2.waitKey(1) & 0xFF
            
            if (key == ord("m")):
                # print(self.tello.get_bat())
                dst,mask = self.preproccessAndKey(frame)
                # rect =
                if(self.trigger_init_start==0):
                    # print("tfadygsuhijfsfjhtgdyuhjkdfgytucijcdfgyusijcdf")
                    
                    rect = self.get_coordinates(mask,dst)
                    print(rect)
                    if(rect[0][0] == 0):
                        self.rcOut[2] = 20              #changed
                        self.rcOut[0] = 0
                        self.rcOut[1] = 0
                        self.rcOut[3] = 0
                        # continue
                    else:
                        print("hahahah")
                        self.trigger_init_start = 1

                if(self.trigger_init_start==1):
                    self.align_rect.run()
                    self.trigger_init_start = 2
                    self.align_rect.clear()
                    # print("gfcxgfhjkhgcfxdfghfxdghjcftyghxfgfhvfxdfgh")


                if(self.trigger_init_start==2):

                    if(up==0):
                        print("up was 0 so going up")
                        # print(self.trigger_init)

                        rect = self.get_coordinates(mask,dst)

                        if(self.trigger_init==0):
                            
                            if(rect[0][0] == 0):
                                # self.rcOut[0] = 0
                                # self.rcOut[1] = 0
                                # self.rcOut[2] = -10
                                # self.rcOut[3] = 0
                                continue
                            else:
                                print("ahhahaha")
                                self.trigger_init = 1

                        if(self.trigger_init == 1):
                            self.align_rect.run()
                            self.trigger_init = 2
                            self.align_rect.clear()

                        if(self.trigger_init == 2):
                            ok = self.start_tracking(rect,dst)
                            if (ok==True):
                                self.trigger_init = 3

                        if(self.trigger_init == 3):
                            self.track_up(dst)
                            if self.trigger == 1:
                                # up = 1
                                self.trigger = 0
                                self.trigger_init = 4

                        if(self.trigger_init == 4):
                            # rect = self.get_coordinates(mask,dst)
                            if(rect[0][0] == 0):
                                self.rcOut[2] = 20
                                self.rcOut[0] = 0
                                self.rcOut[1] = 0
                                self.rcOut[3] = 0 
                            else:
                                print("hahahah")
                                self.trigger_init = 5

                        if(self.trigger_init == 5):

                            self.align_rect.run()
                            self.trigger_init = 0
                            up = 1
                            self.align_rect.clear()

                    elif (left!=0):

                        print("now we are up and now going right")
                        rect = self.get_coordinates(mask,dst)

                        if(self.trigger_init==0):
                            # rect = self.get_coordinates(mask,dst)
                            if(rect[0][0] == 0):
                                continue
                            else:
                                print("ahhahaha")
                                self.trigger_init = 1

                        if(self.trigger_init == 1):
                            self.align_rect.run()
                            self.trigger_init = 2
                            self.align_rect.clear()

                        if(self.trigger_init == 2):
                            ok = self.start_tracking(rect,dst)
                            if(ok==True):
                                self.trigger_init = 3

                        if(self.trigger_init == 3):
                            self.track_left(dst)
                            if self.trigger == 1:
                                self.trigger = 0
                                self.trigger_init = 4

                        if(self.trigger_init == 4):
                            rect = self.get_coordinates(mask,dst)
                            if(rect[0][0] == 0):
                                self.rcOut[2] = 0
                                self.rcOut[0] = 20   #changed
                                self.rcOut[1] = 0
                                self.rcOut[3] = 0 
                            else:
                                print("hahahah")
                                self.trigger_init = 5

                        if(self.trigger_init == 5):

                            self.align_rect.run()
                            self.trigger_init = 0
                            left = left -1
                            self.align_rect.clear()

                    if left == 0 and up == 1:
                        print("wohoo we have reached where we wanted to be")
                        break

            else:
                self.manualRcControl(key)

            self.sendRcControl()

    def clear(self):
        
        # self.cap = cv2.VideoCapture(0)
        self.tracker = cv2.TrackerKCF_create()
        # self.tracker = cv2.CSRT_create()
        self.rcOut = np.zeros(4)
        self.bbox = (5,5,20,20)

        self.trigger = 0
        self.trigger_init_start = 0

        self.trigger_init = 0

        self.ar = 0

        self.ARmean = np.array([0])
        self.ARqueue = np.zeros((7,1))
        self.ARvar = np.array([0])

        self.lost = 0

        self.visible = 0

        self.align_rect = align_rect(self.tello)   


    def manualRcControl(self,key):
        if key == ord("w"):
            self.rcOut[1] = 50
        elif key == ord("a"):
            self.rcOut[0] = -50
        elif key == ord("s"):
            self.rcOut[1] = -50
        elif key == ord("d"):
            self.rcOut[0] = 50
        elif key == ord("u"):
            self.rcOut[2] = 50
        elif key == ord("j"):
            self.rcOut[2] = -50
        elif key == ord("l"):
            self.tello.land()
        elif key == ord("t"):
            try:
                self.tello.takeoff()
            except:
                print("takeoff toh ho gya lol")
            time.sleep(2)
        else:
            self.rcOut = [0,0,0,0]
        return

    def sendRcControl(self):
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

        frame = cv2.GaussianBlur(frame, (7,7), 0)
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3,3), np.uint8)

        frame_threshold = cv2.inRange(frame_HSV, (19, 0, 176), (64, 37, 255))

        frame_threshold = cv2.dilate(frame_threshold, kernel, iterations=7)
        frame_threshold = np.repeat(frame_threshold[:, :, np.newaxis], 3, 2)

        frame2 = np.uint8((frame_threshold//255)*np.int64(frame_HSV))

        frame_threshold = cv2.inRange(frame2, (20, 28, 73), (57, 139, 133))

        mask = frame_threshold
        kernel = np.ones((1,1), np.uint8)      

        mask = cv2.erode(mask, kernel, iterations=5)

        mask = cv2.dilate(mask, kernel, iterations=8)
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
            if area > 300:#param

                if len(approx) == 4:
                    cnt = approx
                 
                    ar = (np.linalg.norm(approx[0] - approx[1]) + np.linalg.norm(approx[2] - approx[3]))/(np.linalg.norm(approx[2]-approx[1])+np.linalg.norm(approx[0]-approx[3]))
                    if ar > 1:
                        ar=1/ar
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area)/hull_area

                    condition = ar < 0.4 and ar > 0.25
                    if solidity > 0.95 and condition:

                        self.ar = ar
                        # print "ar",self.ar

                        self.ARqueue = np.roll(self.ARqueue,1,axis = 0)
                        self.ARqueue[0,:] = [ar]

                        self.ARvar = np.var(self.ARqueue,axis=0)
                        self.ARmean = np.mean(self.ARqueue,axis = 0)

                        if area > oldArea:
                            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                            #cv2.circle(frame,(int(cx),int(cy)), 3, (0,0,255), -1)
                            cv2.putText(frame, "Rectangle", (x, y), font, 1, (0, 0, 0))

                            cntMain = approx
                            rect = self.order_points(cntMain)
                            # print("reached here")

                        oldArea = area

        return rect

    def start_tracking(self, rect,frame):

        self.bbox = (rect[0][0],rect[0][1],rect[2][0]-rect[0][0],rect[2][1]-rect[0][1])
        self.tracker = cv2.TrackerKCF_create()
        ok = self.tracker.init(frame,self.bbox)
        # p1 = (int(self.bbox[0]), int(self.bbox[1]))
        # p2 = (int(self.bbox[0]+ self.bbox[2]), int(self.bbox[1]+self.bbox[3]))
        # cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        # cv2.imshow("with frame",frame)
        # cv2.waitKey(10)
        return ok

    def track_left(self,frame):

        ok, self.bbox = self.tracker.update(frame)

        if ok:
            p1 = (int(self.bbox[0]), int(self.bbox[1]))
            p2 = (int(self.bbox[0]+ self.bbox[2]), int(self.bbox[1]+self.bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.imshow("with frame",frame)
            # cv2.waitKey(10)
            print("still visible")
            self.rcOut[0] = 20
            self.rcOut[1] = 0
            self.rcOut[2] = 0
            self.rcOut[3] = 0
            self.trigger = 0
            self.lost = 0

        else:
            print("LOST")
            # self.lost +=1
            # if(self.lost>15):
            self.trigger = 1
                # self.lost = 0


    def track_up(self,frame):

        ok, self.bbox = self.tracker.update(frame)

        if ok:
            # p1 = (int(self.bbox[0]), int(self.bbox[1]))
            # p2 = (int(self.bbox[0]+ self.bbox[2]), int(self.bbox[1]+self.bbox[3]))
            # cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            # cv2.imshow("with frame",frame)
            print("still visible")
            self.rcOut[0] = 0
            self.rcOut[1] = 0
            self.rcOut[2] = 40
            self.rcOut[3] = 0
            self.trigger = 0
            self.lost = 0
            self.visible+=1

        else:
            if(self.visible<5):
                self.trigger_init = 2
                return
            print("LOST")
            self.visible = 0
            # self.lost +=1
            # if(self.lost>15):
            self.trigger = 1
                # self.lost = 0

def main():
    print("ahahah")
    tello = Tello()
    tello.connect()
    tello.streamoff()
    tello.streamon()
    frontend = FrontEnd(tello)
    print("noooo")

    frontend.run(1,0, tello.get_yaw())    #left,up

    print("found the end of the code")

if __name__ == '__main__':
    main()