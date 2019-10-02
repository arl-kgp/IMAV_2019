def PoseEstimation(rect):

    K = np.array([[6.981060802052014651e+02,0.000000000000000000e+00,3.783628172155137577e+02],[0.000000000000000000e+00,6.932839845949604296e+02,2.823973488087042369e+02],[0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
    # dist = np.array([-1.428750372096417864e-01,-3.544750945429044758e-02,1.403740315118516459e-03,-2.734988255518019593e-02,1.149084393996809700e-01])

    # K = np.array([[6.331284731799049723e+02,0.000000000000000000e+00,3.240546706735938187e+02],[0.000000000000000000e+00,6.276117931324869232e+02,2.404437048001034611e+02],[0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
    K_inv = np.linalg.inv(K)
    crn = rect

    # print "crn",crn
    # crnVect = np.array([[crn[0]],[crn[1]],[1]])

    crnList = rect

    src = np.array([[-10,-37.5],[10,-37.5],[10,37.5],[-10,37.5]])

    h, status = cv2.findHomography(src,crnList)

    det = np.linalg.det(h)

    if det != 0 :

        hInv = np.linalg.inv(h)

        # o = np.matmul(hInv,crnVect)

        # o = o/o[2]
        # print "o",o

        # h = np.matmul(K,h)
        # print "homography by apriltag", H
        # print "homography by me", h

                

        h1h2h3 = np.matmul(K_inv,h)

        h1T = h1h2h3[:,0]
        h2T = h1h2h3[:,1]
        h3T = h1h2h3[:,2]
        

        h1Xh2T = np.cross(h1T,h2T)


        h1_h2_h1Xh2T = np.array([h1T,h2T,h1Xh2T])
        h1_h2_h1Xh2 = np.transpose(h1_h2_h1Xh2T)

        u, s, vh = np.linalg.svd(h1_h2_h1Xh2, full_matrices=True)

        uvh = np.matmul(u,vh)
        det_OF_uvh = np.linalg.det(uvh)

        M = np.array([[1,0,0],[0,1,0],[0,0,det_OF_uvh]])

        T = h3T/np.linalg.norm(h1T) # Translation Matrix
        T = T*100/17.5
        r = np.matmul(u,M)
        R = np.matmul(r,vh) # Rotation matrix
        T = T

        T_t = np.reshape(T,(3,1))
        neg_Rt_T = -1*np.dot(R.T,T_t)
        f = np.array([[0,0,0,1]])

        
        if neg_Rt_T[2,0] < 0:
            flag = -1
        else:
            flag = 1

        neg_Rt_T[2,0] = neg_Rt_T[2,0]*flag
        neg_Rt_T[0,0] = neg_Rt_T[0,0]*(-1)
        Pose = neg_Rt_T.T
        EulerAngles = rotationMatrixToEulerAngles(R.T)

        # print "Pose",Pose
        # print "pLocal",pLocal
        mat = rotationMatrixToEulerAngles(R)

        roll = mat[0]
        pitch =  mat[1]
        yaw = mat[2]

    else:
        Pose = np.array([[0,0,0]])
        pass

    return Pose


def order_points(pts):

    pts = pts.reshape(4,2)
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    print "dim",pts.shape
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


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def nothing(x):
    # any operation
    pass

import apriltag
from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
import imutils as im

# Speed of the drone
S = 60
# Frames per second of the pygame window display
FPS = 25

font = cv2.FONT_HERSHEY_COMPLEX


class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()
        self.telloPose = np.zeros((1,3))
        self.telloEulerAngles = np.zeros((1,3))

        self.rcOut=np.zeros(4)
        

        self.poseQueue = np.zeros((7,3))
        self.supreQueue = np.zeros((7,3))

        self.flag = 0
        self.telloPoseVariance = np.zeros(3)
        self.telloPoseMean = np.zeros(3)
        self.tello.TIME_BTW_RC_CONTROL_COMMANDS = 20

        self.R = np.zeros((3,3))

        # self.telloPose = np.array([])
            # self.telloEulerAngles = EulerAngles

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()

        should_stop = False

        Height = 100
        while not should_stop:
            if frame_read.stopped:
                frame_read.stop()
                break


            frameBGR = np.copy(frame_read.frame)
            # frameBGR = np.rot90(frameBGR)
            # frameBGR = np.flipud(frameBGR)
            frame2use = im.resize(frameBGR,width=720)
            
            frame = frame2use 
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


            cv2.imshow("fff",frame)
            mask = maskSab

            # Contours detection
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            oldArea = 300
            for cnt in contours:
                area = cv2.contourArea(cnt)
                approx = cv2.approxPolyDP(cnt, 0.012*cv2.arcLength(cnt, True), True) # 0.012 param
                x = approx.ravel()[0]
                y = approx.ravel()[1]

                

                if area > 300:#param

                    # if len(approx) == 3:
                        # cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))
                    if len(approx) == 4:
                        (cx,cy),(MA,ma),angle = cv2.fitEllipse(cnt)
                        ar = MA/ma


                        # area = cv2.contourArea(cnt)
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area)/hull_area

                        # print "Angle",angle
                        # print "solidity",solidity
                        # print "ar",ar
                        if solidity > 0.9 and ar < 0.4:

                            if area > oldArea:
                                cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                                cv2.circle(frame,(int(cx),int(cy)), 3, (0,0,255), -1)
                                cv2.putText(frame, "Rectangle" + str(angle), (x, y), font, 1, (0, 0, 0))

                                cntMain = approx
                                rect = order_points(cntMain)
                                print "rect",rect
                                

                                Pose = PoseEstimation(rect)

                                pX = Pose[0,0]
                                pY = Pose[0,1]
                                pZ = Pose[0,2]

                                Pose[0,0] = pZ
                                Pose[0,1] = -pX
                                Pose[0,2] = -pY


                                self.telloPose = np.transpose(Pose)

                                self.poseQueue = np.roll(self.poseQueue,1,axis = 0)
                                self.poseQueue[0,:] = [Pose[0,0],Pose[0,1],Pose[0,2]]

                                self.telloPoseVariance = np.var(self.poseQueue,axis=0)
                                self.telloPoseMean = np.mean(self.poseQueue,axis = 0)

                                self.flag = 1
                                # print "PoseQueue",self.poseQueue
                                print "PoseMean",self.telloPoseMean
                                # print "telloPoseVariance" , self.telloPoseVariance

                                varN = np.linalg.norm(self.telloPoseVariance)
                                print "varN",varN
                            oldArea =area
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF;

            
            K = np.array([[6.981060802052014651e+02,0.000000000000000000e+00,3.783628172155137577e+02],[0.000000000000000000e+00,6.932839845949604296e+02,2.823973488087042369e+02],[0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
            dist = np.array([-1.128288663079663086e-02,1.551794079596884035e-02,3.003426614702892333e-03,1.319203673619398672e-03,1.086713281720452368e-01])

            K_inv = np.linalg.inv(K)

            h , w = frame2use.shape[:2]

            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

            mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
            dst = cv2.remap(frame2use,mapx,mapy,cv2.INTER_LINEAR)

            x,y,w,h = roi
            dst = dst[y:y+h,x:x+w]
            # print("ROI: ",x,y,w,h)

            cv2.imshow("Orignal",frame2use)
            cv2.imshow("rectified",dst)
            # gray = cv2.cvtColor(frame2use, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
           


            if self.flag == 1:
                varN = np.linalg.norm(self.telloPoseVariance)
                # print "varN",varN
                Pose = self.telloPoseMean


                xEr = 450 - Pose[0]   
                yEr = 0 - Pose[1]
                zEr = 0 - Pose[2]
                ErrorN = np.linalg.norm([xEr,yEr,zEr])

                if key == ord("e"):

                    kp = 0.35
                    MtnCmd = np.array([kp*xEr,kp*yEr,kp*zEr])

                    MtnCmd[0] = -1*MtnCmd[0]
                    self.rcOut = [MtnCmd[1], MtnCmd[0],MtnCmd[2],0]
                    

                    if self.rcOut[0] > 35:
                        self.rcOut[0] = 35
                    elif self.rcOut[0] < -35:
                        self.rcOut[0] = -35

                    if self.rcOut[1] > 35:
                        self.rcOut[1] = 35
                    elif self.rcOut[1] < -35:
                        self.rcOut[1] = -35

                    if self.rcOut[2] > 35:
                        self.rcOut[2] = 35
                    elif self.rcOut[2] < -35:
                        self.rcOut[2] = -35
                else :
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
                    else:
                        self.rcOut = [0,0,0,0]



            else:
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
                else:
                    self.rcOut = [0,0,0,0]

            # print self.rcOut
            self.tello.send_rc_control(int(self.rcOut[0]),int(self.rcOut[1]),int(self.rcOut[2]),int(self.rcOut[3]))
            self.rcOut = [0,0,0,0]

            if key == ord("q"):
                break
            if key == ord("t"):
                self.tello.takeoff()    
            if key == ord("l"):
                self.tello.land()
                Height = 100

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1 / FPS)
            self.flag = 0

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()

