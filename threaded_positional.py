import threading
from djitellopy import Tello
from time import sleep,time
from math import radians, sin, cos
import numpy as np
import cv2


class get_pose:

    def __init__(self):

        self.tello = Tello()
        self.tello.connect()
        sleep(3)
        print("tello connected")
        self.px = 0
        self.py = 0
        self.vx = 0
        self.vy = 0
        # while(self.tello.get_yaw)
        self.initial_yaw = self.tello.get_yaw()
        self.initial_pitch = self.tello.get_pitch()
        self.initial_roll = self.tello.get_roll()
        self.initial_ax = self.tello.get_agx()
        self.initial_ay = self.tello.get_agy()
        self.initial_az = self.tello.get_agz()
        
        self.tello.streamoff()
        self.tello.streamon()

        thread1 = threading.Thread(target = self.extract, args=())
        thread1.daemon = True
        thread1.start()

    def get_acc(self):

        roll = radians(self.tello.get_roll()-self.initial_roll)
        pitch = radians(self.tello.get_pitch()-self.initial_pitch)
        yaw = -radians(self.tello.get_yaw()-self.initial_yaw)

        Tz = np.matrix([[cos(yaw), sin(yaw), 0], [-sin(yaw), cos(yaw), 0], [0, 0, 1]])
        Ty = np.matrix([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
        Tx = np.matrix([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])

        T_n = Tz*Ty*Tx

        A = np.matrix([[self.tello.get_agx()], [self.tello.get_agy()], [self.tello.get_agz()]])

        A_correct = T_n*A

        A_correct[0] = A_correct[0]-self.initial_ax
        A_correct[1] = A_correct[1]-self.initial_ay
        A_correct[2] = A_correct[2]-self.initial_az

        return A_correct

    def extract(self):

        while True:
            start = time()
            A = self.get_acc()
            ax = A[0]
            ay = A[1]
            end = time()

            vx1 = ax*(end-start)
            vy1 = ay*(end-start)

            self.vx = self.tello.get_vgx() + vx1
            self.vy = self.tello.get_vgy() + vy1

            # self.vx = self.vx + vx1
            # self.vy = self.vy + vy1

            self.px = self.px + self.vx*(end-start)
            self.py = self.py + self.vy*(end-start)
            sleep(0.01)

    def run(self):

        frame_read = self.tello.get_frame_read()
        rcOut = [0,0,0,0]
        i=0
        while True:
            if (i%20) == 0:
                print("X position is {} , Y position is {}".format(self.px,self.py))
                print("Correct acceleration is {} ".format(self.get_acc()))
                # print("{}         {}           {}".format(self.initial_ax,self.initial_ay,self.initial_az))
            i = i+1

            frame = frame_read.frame
            cv2.imshow("frame",frame)

            k = cv2.waitKey(2) & 0xFF

            if k == ord("t"):
                self.tello.takeoff()
                sleep(3)
            elif k == ord("l"):
                self.tello.land()
            elif k == ord("w"):
                rcOut[1] = 50
            elif k == ord("a"):
                rcOut[0] = -50
            elif k == ord("s"):
                rcOut[1] = -50
            elif k == ord("d"):
                rcOut[0] = 50
            elif k == ord("u"):
                # print("up")
                rcOut[2] = 50
            elif k == ord("j"):
                rcOut[2] = -50
            elif k == ord("c"):
                rcOut[3] = 50
            elif k == ord("v"):
                rcOut[3] = -50
            elif k == ord("q"):
                self.tello.land()
                self.tello.end()
                break

            # self.tello.send_rc_control(int(rcOut[0]),int(rcOut[1]),int(rcOut[2]),int(rcOut[3]))
            rcOut = [0,0,0,0]


def main():
    position = get_pose()
    position.run()
    # position.extract()

if __name__ == '__main__':
    main()