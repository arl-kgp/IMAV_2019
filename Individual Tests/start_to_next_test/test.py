import cv2
import numpy as np
cap = cv2.VideoCapture("test.avi")

def rectifyInputImage(self,frame2use):

    K = np.array([[7.092159469231584126e+02,0.000000000000000000e+00,3.681653710406367850e+02],[0.000000000000000000e+00,7.102890453175559742e+02,2.497677007139825491e+02],[0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])          #to change camera parameters
    dist = np.array([2.439122447395965926e-02,-1.174125872015051447e-01,-7.226737851943197850e-03,-2.109186754013973528e-03,6.156184110527554987e-01])                                                                                                              #to change distortion coefficients
    K_inv = np.linalg.inv(K)

    h , w = frame2use.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

    mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
    dst = cv2.remap(frame2use,mapx,mapy,cv2.INTER_LINEAR)

    x,y,w,h = roi
    dst = dst[y:y+h,x:x+w]

    return dst

def getRectMask(self,frame):
    kernel = np.ones((5,5),np.uint8)#param 1

    blurred = cv2.GaussianBlur(frame, (7, 7), 0)#param 1

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    dilS = cv2.dilate(s,kernel,iterations = 1)
    newS = dilS-s
    newS = cv2.equalizeHist(newS)
    # newS = cv2.GaussianBlur(newS, (11, 11), 0)


    dilV = cv2.dilate(v,kernel,iterations = 1)#param 1
    newV = dilV-v
    newV = cv2.equalizeHist(newV)

    dilH = cv2.dilate(h,kernel,iterations = 1)
    newH = dilH-h
    newH = cv2.equalizeHist(newH)


    sabKaAnd = cv2.bitwise_or(newS,newV)
    kernel2 = np.ones((3,3),np.uint8)#param 1
    sabKaAnd = cv2.erode(sabKaAnd,kernel2,iterations = 1)#param 1
    sabKaAnd = cv2.erode(sabKaAnd,kernel2,iterations = 1)#param 1

    sabKaAnd = cv2.dilate(sabKaAnd,kernel2,iterations = 1)#param 1
    sabKaAnd = cv2.GaussianBlur(sabKaAnd, (11, 11), 0)

    maskSab = cv2.inRange(sabKaAnd,120,255)#param 1****

    maskSab = cv2.erode(maskSab,kernel2,iterations = 1)
    maskSab = cv2.dilate(maskSab,kernel2,iterations = 1)

    maskSab = cv2.bitwise_and(maskSab,newV)
    maskSab = cv2.equalizeHist(maskSab)
    maskSab = cv2.inRange(maskSab,190,255)# param *****

    kernel2 = np.ones((2,2),np.uint8) #param ****
    maskSab = cv2.erode(maskSab,kernel2,iterations = 1)
    maskSab = cv2.dilate(maskSab,kernel2,iterations = 1)

    return maskSab

def preproccessAndKey(frame_read):
    frameBGR = np.copy(frame_read.frame)
    frame2use = im.resize(frameBGR,width=720)
        
    frame = frame2use 

    key = cv2.waitKey(1) & 0xFF;

    dst = self.rectifyInputImage(frame2use)            
    mask = self.getRectMask(dst)

    return key,dst,mask

def PoseEstimationfrmMask(self,mask,frame,frameH,frameW,arSet):
    # Contours detection
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    oldArea = 300                                                                                                
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.012*cv2.arcLength(cnt, True), True)                                     #to change 0.012 param IMP
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        

        if area > 300:#param                                                                                     #to change
            # if len(approx) == 3:
                # cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))
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

                # print "Angle",angle
                # print "solidity",solidity
                # print "ar",ar
                condition = ar < 1 and ar > arSet
                if solidity > 0.9 and condition:                                                                #to change

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
                        # print "rect",rect
                        
                        Pose = self.PoseEstimation(rect,frameH,frameW)
                        if self.PoseFlag == 1:
                            # print "PoseFlag",self.PoseFlag
                            self.telloPose = np.transpose(Pose)

                            self.poseQueue = np.roll(self.poseQueue,1,axis = 0)
                            self.poseQueue[0,:] = [Pose[0,0],Pose[0,1],Pose[0,2]]

                            self.telloPoseVariance = np.var(self.poseQueue,axis=0)
                            self.telloPoseMean = np.mean(self.poseQueue,axis = 0)
                            # print "PoseQueue",self.poseQueue
                            # print "PoseMean",self.telloPoseMean
                            # print "telloPoseVariance" , self.telloPoseVariance
                        else:
                            pass

                        varN = np.linalg.norm(self.telloPoseVariance)
                        # print "varN",varN
                    oldArea =area

while(1):

	_,frame = cap.read()
	key = cv2.waitKey(5) & 0xFF

	key,dst,mask = preproccessAndKey(frame)

    frameH,frameW,arSet = 40,40,0.8                                              #to change (might)
    cv2.imshow("msk",mask)
    PoseEstimationfrmMask(mask,dst,frameH,frameW,arSet)