from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils as im
from toffNsrchShelf import FrontEnd as align_initial
from skip_shel import FrontEnd as skip_first
from orient_yaw import Orient as orient
# import xbox

class starting(object):

    def __init__(self):

        self.tello = Tello()
        # self.joy = xbox.Joystick()
        self.tello.connect()
        self.tello.streamoff()
        self.tello.streamon()
        self.orient = orient(self.tello)
        self.align_initial = align_initial(self.tello)
        self.skip_first = skip_first(self.tello)
        self.align_next = align_initial(self.tello)

    def run(self):

        # takeoff
        # self.tello.takeoff()
        # time.sleep(2)

        # self.orient.orient(5)
        # yaw orientation correct
        # go to height
        # read flag
        # rotate cloclwise/anti 180

        print("mission started")
        self.align_initial.run()
        print("Now have aligned in front of first frame")
        self.skip_first.run()
        print("skipped the first frame")
        self.align_next.run()
        print("Now have aligned in front of second frame")

        # go more distance now

        # do warehouse

        # search for the shelf to pass from

        # 


        self.tello.land()
        print("Ended")

    # def controller(self):
        
    #     while(True): 
    #         leftStick = joy.leftStick()
    #         rightStick = joy.rightStick()
    #         self.keydown("leftst",leftStick)
    #         self.keydown("rightst",rightStick)

    #         if joy.Y():
    #             self.keyup("takeoff")
    #         if joy.B():
    #             self.keyup("land")

    #         if joy.X() and joy.leftBumper() and joy.rightBumper():          #killswitch
    #             self.keyup("emergency")
    #         if frame_read.stopped:
    #             frame_read.stop()
    #             break
            
    #         if joy.A():
    #             self.keyup("land")
    #             should_stop = True
    #         self.update()

    #         time.sleep(0.5)

    # def keydown(self, key, value):
    #     """ Update velocities based on key pressed
    #     Arguments:
    #         key: pygame key
    #     """
    #     if key == "rightst":  # set roll/pitch velocity
    #         self.for_back_velocity = int(value[1]*S)         #pitch
    #         self.left_right_velocity = int(value[0]*S)       #roll
    #     elif key == "leftst": #set yaw/throttle
    #         self.yaw_velocity = int(value[0]*S)              #yaw
    #         self.up_down_velocity = int(value[1]*S)          #throttle

    # def keyup(self, key):
    #     """ Update velocities based on key released
    #     Arguments:
    #         key: pygame key
    #     """

    #     if key == "takeoff":  # takeoff
    #         self.tello.takeoff()
    #         self.send_rc_control = True
    #     elif key == "land":  # land
    #         self.tello.land()
    #         self.send_rc_control = False
    #     elif key == "emergency":
    #         self.tello.emergency()
    #         self.send_rc_control = False

    # def update(self):
    #     """ Update routine. Send velocities to Tello."""
    #     if self.send_rc_control:
    #         self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
    #                                    self.yaw_velocity)


def main():
    print("now i am gonna start the mission")
    start = starting()
    start.run()

if __name__ == '__main__':
    main()


