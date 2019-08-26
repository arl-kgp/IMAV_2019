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
FPS = 20


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


        self.lk_params = dict( winSize  = (15,15),
                                  maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1.0))
        self.m = 1

        self.frameCols = 320
        self.frameRows = 180
        self.p0 = np.array([])

        # self.telloPose = np.array([])
            # self.telloEulerAngles = EulerAngles
    def run(self):


        self.startStream()
        frame_read = self.tello.get_frame_read()

        should_stop = False

        valStrt1 = 0 
        old_gray  = np.zeros((180,320),dtype = np.uint8)

        diff1 = np.zeros((self.frameRows,self.frameCols),dtype = np.int)
        diff2 = np.zeros((self.frameRows,self.frameCols),dtype = np.int)
        img1 = np.zeros((self.frameRows,self.frameCols),dtype = np.int)
        img2 = np.zeros((self.frameRows,self.frameCols),dtype = np.int)
        old_line = None
        while not should_stop:
            if frame_read.stopped:
                frame_read.stop()
                break
            frameBGR = np.copy(frame_read.frame)
            frame_gray = cv2.cvtColor(frameBGR,cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.resize(frame_gray, (self.frameCols,self.frameRows))
            ################################################################################################################

            img1 = frame_gray/5
            # print "img1",img1.shape,"img2",img2.shape
            diff1 = img1 - img2 
            diff2 = img2 - img1 
            diff1 = cv2.bitwise_or(diff1,diff2)
            diff1 = cv2.inRange(diff1,250,255)
            diff1 = np.array(diff1,dtype=np.uint8)
            kernel = np.ones((3,3),np.uint8)
            erosion = cv2.erode(diff1,kernel,iterations = 1)
            kernel = np.ones((2,2),np.uint8)
            erosion = cv2.dilate(diff1,kernel,iterations = 1)
            img2 = img1
            # new = cv2.bitwise_and(erosion,img)
            # print row,col
            x = self.printcheckboard()
            new1 = cv2.bitwise_and(x,erosion)

            cv2.imshow("new",new1)

            mask1 = np.zeros((self.frameRows,self.frameCols),dtype=np.uint8)
            # print "p0",self.p0


            if(self.p0.size != 0):

                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, self.p0, None, **self.lk_params)
                # Select good points

                good_new = p1[st==1]
                good_old = self.p0[st==1]

                depthMap = np.zeros((self.frameRows,self.frameCols),dtype=np.uint8)

                good_newInt = good_new.astype(int)
                good_oldInt = good_old.astype(int)

                disp = np.absolute(good_newInt-good_oldInt)

                u = 0
                for x in good_newInt:
                    if x[1] <self.frameRows and x[0] <self.frameCols:
                        # print "XXXXXXXX",x,"disp",disp[u,0],"u",u
                        # dis1 = np.linalg.norm([disp[u,0] + disp[u,1]])
                        pass

                        mask1[x[1],x[0]] = disp[u,0]
                    else:
                        pass
                    u = u+1 
                good_new = p1
                good_old = self.p0

                cv2.imshow("mask1",mask1)#mask1 has disparity


                mask1 = cv2.dilate(mask1,kernel,iterations = 1)
                mask1 = cv2.GaussianBlur(mask1,(9,9),0)
                rowMat = np.mean(mask1,axis = 0)
                rowVar = np.var(mask1,axis = 0)

                arr1 = rowVar
                arr1 = arr1 - arr1.mean()
                arr1 = arr1 / arr1.max()


                arr = rowMat
                arr = arr - arr.mean()
                arr = arr / arr.max()

                # plt.plot(arr)
                # plt.plot(arr1)

                arr = rowMat
                arr = arr - arr.mean()
                arr = arr

                # a = arr - arr1
                a= arr
                a = a - a.mean()
                a = a / 5
                aD = a>0
                aD = aD*1
                a = aD*a

                a = a - a.mean()
                a = a / 5
                aD = a>0
                aD = aD*1
                a = aD*a

                a = a - a.mean()
                a = a / 5
                aD = a>0
                aD = aD*1
                a = aD*a

                a =a*10

                lineMask = a*255
                lineMask = lineMask.astype(dtype = np.uint8)
                lineMask2 = cv2.inRange(lineMask,90,255)/255#array with pole maxima
                # print "varLine",varLine,"meanLine",meanLine
                dadgMask = np.repeat(lineMask2[np.newaxis, :], 180, axis=0)
                
                dande = []

                pi = -1
                for ind, v in enumerate(lineMask2[:-1]): 
                    if v[ind] >0 and pi != -1:
                        pi = ind
                    if v[ind] >0 and v[ind+1]==0:
                        dande.append((pi, ind+1))
                        pi = -1
                if lineMask2[-1] >0:
                    if pi != -1:
                        dande.append((pi, len(lineMask2)))
                    else:
                        dande.append((len(lineMask2-1), len(lineMask2)))

                maxd = 0
                sump = np.sum(mask1, axis = 1)/180
                for d in dande:
                    meand = np.sum(sump[d[0]:d[1]])/(d[1]-d[0])
                    if abs(meand) > abs(maxd):
                        maxd = meand
                


                if maxd > 0:
                    lineMask3 = np.concatenate((lineMask2[maxd:], np.zeros(maxd)))
                else:
                    lineMask3 = np.concatenate((np.zeros(-maxd), lineMask2[:maxd]))

                if oldLine is not None:
                    lineMaskF = lineMask3 & oldLine
                    dadgMask = np.repeat(lineMaskF[np.newaxis, :], 180, axis=0)

                oldLine = lineMask3
                kernel = np.ones((4,4),np.uint8)

                # dadgMask = cv2.dilate(dadgMask,kernel,iterations = 1)
                cv2.imshow("lineMask",dadgMask)


            ################################################################################################################
            old_gray = frame_gray
            if(self.m%3 == 0):
                # print "dddddddddddddddddddddddddddd"
                # while self.p0.size == 0:
                
                self.p0 = self.locateFeatures(new1)
                pass

            # else:
                # p0 = good_new.reshape(-1,1,2)
            self.m=self.m+1          


            key = cv2.waitKey(1) & 0xFF;
            # else:
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
                self.rcOut = [-20,0,0,0]

            # print self.rcOut
            self.tello.send_rc_control(int(self.rcOut[0]),int(self.rcOut[1]),int(self.rcOut[2]),int(self.rcOut[3]))
            # self.rcOut = [0,0,0,0]

            if key == ord("q"):
                break
            if key == ord("t"):
                self.tello.takeoff()    
            if key == ord("l"):
                self.tello.land()
                Height = 100

           
            time.sleep(1 /FPS)

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def printcheckboard(self): 
        # create a n * n matrix 
        x = np.zeros((self.frameRows,self.frameCols), dtype = np.uint8) 
        # x = x +255
        # fill with 1 the alternate rows and columns 
        x[1::2, ::4] = 255
        x[::4, 1::2] = 255
        
        return x

    def locateFeatures(self,inImg):
        cv2.imshow("totally new",inImg)
        (rowsLoc , colLoc) = np.where(inImg == 255)   
        rowsLoc = np.reshape(rowsLoc,(-1,1))
        colLoc = np.reshape(colLoc,(-1,1))
        j = np.stack(( colLoc,rowsLoc), axis=-1)
        m = np.array(j,np.float32)
        # print "j",m

        # print "m.shape",f
        return m

    def startStream(self):
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
    



def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()


























# def printcheckboard(r,c): 
#     # create a n * n matrix 
#     x = np.zeros((r, c), dtype = np.uint8) 
#     # x = x +255
#     # fill with 1 the alternate rows and columns 
#     x[1::2, ::2] = 255
#     x[::2, 1::2] = 255
    
#     return x

# def locateFeatures(inImg):
#     cv2.imshow("totally new",inImg)
#     (rowsLoc , colLoc) = np.where(inImg == 255)   
#     rowsLoc = np.reshape(rowsLoc,(-1,1))
#     colLoc = np.reshape(colLoc,(-1,1))
#     j = np.stack(( colLoc,rowsLoc), axis=-1)
#     m = np.array(j,np.float32)
#     # print "j",m

#     # print "m.shape",f
#     return m

# import apriltag
# from djitellopy import Tello
# import cv2
# import pygame
# from pygame.locals import *
# import numpy as np
# import time
# import imutils as im

# # Speed of the drone
# S = 60
# # Frames per second of the pygame window display
# FPS = 25


# class FrontEnd(object):
#     """ Maintains the Tello display and moves it through the keyboard keys.
#         Press escape key to quit.
#         The controls are:
#             - T: Takeoff
#             - L: Land
#             - Arrow keys: Forward, backward, left and right.
#             - A and D: Counter clockwise and clockwise rotations
#             - W and S: Up and down.
#     """

#     def __init__(self):
#         # Init pygame
#         # Init Tello object that interacts with the Tello drone
#         self.tello = Tello()
#         self.telloPose = np.zeros((1,3))
#         self.telloEulerAngles = np.zeros((1,3))

#         self.rcOut=np.zeros(4)
        

#         self.poseQueue = np.zeros((7,3))
#         self.supreQueue = np.zeros((7,3))

#         self.flag = 0
#         self.telloPoseVariance = np.zeros(3)
#         self.telloPoseMean = np.zeros(3)
#         self.tello.TIME_BTW_RC_CONTROL_COMMANDS = 20

#         self.R = np.zeros((3,3))

#         # self.telloPose = np.array([])
#             # self.telloEulerAngles = EulerAngles

#     def run(self):


#         if not self.tello.connect():
#             print("Tello not connected")
#             return

#         # In case streaming is on. This happens when we quit this program without the escape key.
#         if not self.tello.streamoff():
#             print("Could not stop video stream")
#             return

#         if not self.tello.streamon():
#             print("Could not start video stream")
#             return

#         frame_read = self.tello.get_frame_read()

#         should_stop = False

#         m=1


#         # Parameters for lucas kanade optical flow
#         lk_params = dict( winSize  = (15,15),
#                                   maxLevel = 2,
#                                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1.0))


#         valStrt1 = 0 
#         while not should_stop:
#             if frame_read.stopped:
#                 frame_read.stop()
#                 break
#             frameBGR = np.copy(frame_read.frame)

#             frame   = frameBGR
#             K = np.array([[7.092159469231584126e+02,0.000000000000000000e+00,3.681653710406367850e+02],[0.000000000000000000e+00,7.102890453175559742e+02,2.497677007139825491e+02],[0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
#             dist = np.array([-1.428750372096417864e-01,-3.544750945429044758e-02,1.403740315118516459e-03,-2.734988255518019593e-02,1.149084393996809700e-01])
#             K_inv = np.linalg.inv(K)

#             h , w = frame.shape[:2]

#             newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

#             mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
#             dst = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

#             x,y,w,h = roi
#             dst = dst[y:y+h,x:x+w]

#             cv2.waitKey(1)

#             if valStrt1 > 1000:
#                 break
#             valStrt1 = valStrt1+1
#             frame = cv2.resize(dst, (320,180)) 
#             old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             print "old_gray ki size", old_gray.shape
            
#             p0 = locateFeatures(old_gray)        
#             if p0.size != 0:
#                 break

#             cv2.imshow("old_gray",old_gray)
#             cv2.waitKey(1)
#             time.sleep(1 / FPS)

#         # old_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#         # print p0

#         row,col= 180,320
#         print "RowsANdCols",row,col
#         diff1 = np.zeros((row,col),dtype = np.int)
#         diff2 = np.zeros((row,col),dtype = np.int)
#         img1 = np.zeros((row,col),dtype = np.int)
#         img2 = np.zeros((row,col),dtype = np.int)

#         mask = np.zeros((row,col),dtype = np.uint8)
#         while not should_stop:
#             if frame_read.stopped:
#                 frame_read.stop()
#                 break

#             frameBGR = np.copy(frame_read.frame)
#             frame = frameBGR

#             key = cv2.waitKey(1) & 0xFF;
            
#             K = np.array([[7.092159469231584126e+02,0.000000000000000000e+00,3.681653710406367850e+02],[0.000000000000000000e+00,7.102890453175559742e+02,2.497677007139825491e+02],[0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
#             dist = np.array([-1.428750372096417864e-01,-3.544750945429044758e-02,1.403740315118516459e-03,-2.734988255518019593e-02,1.149084393996809700e-01])
#             K_inv = np.linalg.inv(K)

#             h , w = frame.shape[:2]

#             newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

#             mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
#             dst = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)

#             x,y,w,h = roi
#             dst = dst[y:y+h,x:x+w]
#             # print("ROI: ",x,y,w,h)
#             ##########################################################################################################

#             frame = cv2.resize(dst, (320,180)) 
#             image = frame
#             frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # frame_gray = cv2.GaussianBlur(frame_gray,(5,5),0.3)
#             #cv2.imshow("Frame",frame_gray)
            
#             imge = frame_gray
#             img1 = imge/5
#             print "img1",img1.shape,"img2",img2.shape
#             diff1 = img1 - img2 +1
#             diff2 = img2 - img1 +1
#             diff1 = cv2.bitwise_or(diff1,diff2)
            
#             diff1 = np.array(diff1,dtype=np.uint8)
#             kernel = np.ones((3,3),np.uint8)
#             erosion = cv2.erode(diff1,kernel,iterations = 1)
#             kernel = np.ones((2,2),np.uint8)
#             erosion = cv2.dilate(diff1,kernel,iterations = 1)
#             img2 = img1
#             # new = cv2.bitwise_and(erosion,img)
#             # print row,col
#             x = printcheckboard(row,col)
#             new1 = cv2.bitwise_and(x,erosion)

#             cv2.imshow("new",new1)

#             mask1 = np.zeros((row,col),dtype=np.uint8)
#             # calculate optical flow
#             print "p0.size",p0.size
#             if(p0.size != 0):
#                 print "ssssssssssssssssssssssssssssssssssssssssssssssssss"

#                 p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#                 # Select good points

#                 good_new = p1[st==1]
#                 good_old = p0[st==1]

#                 depthMap = np.zeros((row,col),dtype=np.uint8)

#                 good_newInt = good_new.astype(int)
#                 good_oldInt = good_old.astype(int)

#                 disp = np.absolute(good_newInt-good_oldInt)

#                 u = 0
#                 for x in good_newInt:
#                     if x[1] <row and x[0] <col:
#                         # print "XXXXXXXX",x,"disp",disp[u,0],"u",u
#                         # dis1 = np.linalg.norm([disp[u,0] + disp[u,1]])
                        
#                         mask1[x[1],x[0]] = disp[u,0]
#                     else:
#                         pass
#                     u = u+1 
#                 good_new = p1
#                 good_old = p0
#                 kernel = np.ones((5,5),np.uint8)
#                 mask1 = cv2.dilate(mask1,kernel,iterations = 1)
#                 mask1 = cv2.GaussianBlur(mask1,(9,9),0)
#                 mask1 = cv2.equalizeHist(mask1)
#                 rowMat = np.mean(mask1,axis = 0)
#                 rowVar = np.var(mask1,axis = 0)

#                 arr1 = rowVar
#                 arr1 = arr1 - arr1.mean()
#                 arr1 = arr1 / arr1.max()


#                 arr = rowMat
#                 arr = arr - arr.mean()
#                 arr = arr / arr.max()

#                 # plt.plot(arr)
#                 # plt.plot(arr1)

#                 a = arr - arr1
#                 a = a*arr

#                 a = a - a.mean()
#                 a = a / a.max()

#                 aD = a>0
#                 aD = aD*1
#                 aI = aD*a

#                 lineMask = aI*255
#                 lineMask = lineMask.astype(dtype = np.uint8)

#                 varLine = np.var(aI)
#                 meanLine = np.mean(aI)

#                 # print "varLine",varLine,"meanLine",meanLine
#                 dadgMask = np.repeat(lineMask[np.newaxis, :], 180, axis=0)
                
#                 kernel = np.ones((4,4),np.uint8)

#                 dadgMask = cv2.dilate(dadgMask,kernel,iterations = 1)
#                 cv2.imshow("lineMask",dadgMask)
#                 # print "hdhfjkdf",a
#                 # plt.plot(aI)        
#                 # plt.plot(rowVar)
#                 # plt.plot(haba)
#                 # plt.show(1)

#                 dadgMask = cv2.resize(dadgMask, (960,720))
#                 mask1 = cv2.resize(mask1, (960,720))
                

#                 cv2.imshow("masaksjask",mask1)

#             else:
                
#                 continue
#             # img = cv2.add(image,mask)
            
#             # cv2.imshow('frame',img)

#             # Now update the previous frame and previous points
#             old_gray = frame_gray

#             # key = cv2.waitKey(1) & 0xFF
#             if(m%10 == 0):
#                 print "dddddddddddddddddddddddddddd"
#             # if key == ord("e"):    
#                 p0 = locateFeatures(new1)
#                 # locateFeatures(new1)
#                 # print "yes"
#                 pass

#             else:
#                 p0 = good_new.reshape(-1,1,2)
#             m=m+1

#             print "m",m
#             print "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
#             cv2.imshow("Orignal",frame)
#             cv2.imshow("rectified",dst)
            


#             # else:
#             if key == ord("w"):
#                 self.rcOut[1] = 50
#             elif key == ord("a"):
#                 self.rcOut[0] = -50
#             elif key == ord("s"):
#                 self.rcOut[1] = -50
#             elif key == ord("d"):
#                 self.rcOut[0] = 50
#             elif key == ord("u"):
#                 self.rcOut[2] = 50
#             elif key == ord("j"):
#                 self.rcOut[2] = -50
#             else:
#                 self.rcOut = [0,0,0,0]

#             print self.rcOut
#             self.tello.send_rc_control(int(self.rcOut[0]),int(self.rcOut[1]),int(self.rcOut[2]),int(self.rcOut[3]))
#             self.rcOut = [0,0,0,0]

#             if key == ord("q"):
#                 break
#             if key == ord("t"):
#                 self.tello.takeoff()    
#             if key == ord("l"):
#                 self.tello.land()
#                 Height = 100

           
#             time.sleep(1 / FPS)
#             print "m",m

#         # Call it always before finishing. I deallocate resources.
#         self.tello.end()



# def main():
#     frontend = FrontEnd()

#     # run frontend
#     frontend.run()


# if __name__ == '__main__':
#     main()