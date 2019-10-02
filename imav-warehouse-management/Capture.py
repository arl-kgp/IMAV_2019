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

        name = "outImg"
        i = 0

        while not should_stop:
            if frame_read.stopped:
                frame_read.stop()
                break

            frameBGR = np.copy(frame_read.frame)
            # frameBGR = np.rot90(frameBGR)
            # frameBGR = np.flipud(frameBGR)
            frame2use = im.resize(frameBGR,width=720)
            

            key = cv2.waitKey(1) & 0xFF ;

            cv2.imshow("lkgs",frame2use)

    
            if key == ord("c"):
                new = name + str(i) + ".jpg"
                print new
                i = i+1
                cv2.imwrite(new,frame2use)

            key = cv2.waitKey(1) & 0xFF ;
            if key == ord("q"):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)

            # frame = np.rot90(frame)
            # frame = np.flipud(frame)
            # a = frameGray.dtype
            # print a
            
            time.sleep(1 / FPS)

        # Call it always before finishing. I deallocate resources.
        self.tello.end()


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
