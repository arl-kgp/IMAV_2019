from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
import xbox

# Speed of the drone
S = 100
# Frames per second of the pygame window display
FPS = 25
joy = xbox.Joystick()

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
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(USEREVENT + 1, 50)

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
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
        while not should_stop:

            #for event in pygame.event.get():
            #    if event.type == USEREVENT + 1:
            #        self.update()
            #    elif event.type == QUIT:
            #        should_stop = True
            #    elif event.type == KEYDOWN:
            #        if event.key == K_ESCAPE:
            #            should_stop = True
            #        else:
            #            self.keydown(event.key)
            #    elif event.type == KEYUP:
            #        self.keyup(event.key)

            leftStick = joy.leftStick()
            rightStick = joy.rightStick()
            self.keydown("leftst",leftStick)
            self.keydown("rightst",rightStick)

            if joy.Y():
                self.keyup("takeoff")
            if joy.B():
                self.keyup("land")

            if joy.X() and joy.leftBumper() and joy.rightBumper():			#killswitch
            	self.keyup("emergency")
            if frame_read.stopped:
                frame_read.stop()
                break
            
            if joy.A():
                self.keyup("land")
                should_stop = True
            self.update()
            self.screen.fill([0, 0, 0])
            frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)
            self.printer()
        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def keydown(self, key, value):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == "rightst":  # set roll/pitch velocity
            self.for_back_velocity = int(value[1]*S)         #pitch
            self.left_right_velocity = int(value[0]*S)       #roll
        elif key == "leftst": #set yaw/throttle
            self.yaw_velocity = int(value[0]*S)              #yaw
            self.up_down_velocity = int(value[1]*S)          #throttle

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """

        if key == "takeoff":  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == "land":  # land
            self.tello.land()
            self.send_rc_control = False
        elif key == "emergency":
        	self.tello.emergency()
        	self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

    def printer(self):
        print("Yaw: "+str(self.yaw_velocity) + " Throttle: " + str(self.up_down_velocity) + " Pitch: " + str(self.for_back_velocity) + " Roll: " + str(self.left_right_velocity))


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
