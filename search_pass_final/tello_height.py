from djitellopy import Tello
import cv2

import numpy as np

import cv2

from time import sleep

font = cv2.FONT_HERSHEY_SIMPLEX


class FrontEnd(object):

    def __init__(self,tello):
        # Init pygame
        # Init Tello object that interacts with the Tello drone
        self.tello = tello
        self.rcOut=np.zeros(4)
        
       
        self.flag = 0
        self.frame = None

        # self.Height = 200


    def run(self,Height):

        should_stop = False
        land = False
        start_h = False
        off = False

        while not should_stop:

            self.flag = 0
        
            self.rcOut = [0,0,0,0]
            h = self.tello.get_h()
            if not land:

                if abs(Height- h) > 40 :
                    self.rcOut[2] = int((Height - h) * 0.3)
                elif abs(Height- h) > 15:
                    self.rcOut[2] = int((Height - h) * 0.5)
                elif abs(Height- h) <= 15 :
                    if Height == 200:
                        print('300 now!!!!!!!!!!!!!!')
                        Height = 300
                    elif Height == 300:
                        print('100 now!!!!!!!!!!!!!!')
                        Height = 100
                    elif Height == 100:
                        print('200 now!!!!!!!!!!!!!!')
                        Height = 200



            print(h-Height)
            try:
                self.tello.send_rc_control(int(self.rcOut[0]),int(self.rcOut[1]),int(self.rcOut[2]),int(self.rcOut[3]))
            except:
                pass
            self.rcOut = [0,0,0,0]

            sleep(1/60.0)
        # Call it always before finishing. I deallocate resources.


def main():

    tello = Tello()
    tello.connect()
    tello.streamon()
    tello.streamoff()

    frontend = FrontEnd(tello)

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
