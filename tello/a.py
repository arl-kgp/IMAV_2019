from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils as im

# font = cv2.FONT_HERSHEY_COMPLEX

out = cv2.VideoWriter('outpy5.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (720,540))


class FrontEnd(object):

    def __init__(self,tello):
        self.tello = tello
        # self.tello = Tello()

        # self.cap = cv2.VideoCapture(0)
        # self.tracker = cv2.TrackerMedianFlow_create()
        self.rcOut = np.zeros(4)
        self.bbox = (5,5,20,20)

        self.trigger = 0
        self.trigger_init = 0

        self.ar = 0

        self.ARmean = np.array([0])
        self.ARqueue = np.zeros((7,1))
        self.ARvar = np.array([0])

        self.lost = 0

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
        print("4")
        frame_read = self.tello.get_frame_read()

        should_stop = False

        img_array = []

        while not should_stop:
            frame = frame_read.frame
            print(self.tello.get_bat())
            if frame_read.stopped:
                frame_read.stop()
                break

            # print(frame.shape)
            
            # if (key == ord("m")):
            dst = self.preproccessAndKey(frame)

            # img_array.append(dst)
            print(dst.shape)

            out.write(dst)

            cv2.imshow("original",dst)

            key = cv2.waitKey(1) & 0xFF;

            self.manualRcControl(key)

            self.sendRcControl()
            if key == ord("q"):
            	out.release()
            	return

    def preproccessAndKey(self,frame_read):

        frameBGR = frame_read
        frame2use = im.resize(frameBGR,width=720)
            
        frame = frame2use 
        
        dst = frame2use            
        # mask = self.getRectMask(dst)

        return dst

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

def main():
    print("0")
    tello = Tello()
    print("1")
    frontend = FrontEnd(tello)
    print("2")

    frontend.run()

    print("found the end of the code")

if __name__ == '__main__':
    main()
    # return