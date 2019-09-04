from djitellopy import Tello
import cv2
import numpy as np
import time
import imutils as im


from left_right import FrontEnd as left_right
from verticalScanNPass import FrontEnd as up_down
# from passFrmWindow import FrontEnd as passWin
# from after_shelf import FrontEnd as after
from orient_yaw import Orient as orient 

class starting(object):

    def __init__(self,tello):

        self.tello = tello
        # self.joy = xbox.Joystick()
        # self.tello.connect()
        # self.tello.streamoff()
        # self.tello.streamon()
        # print("nooooooooo")

        self.left_right = left_right(self.tello)

        # print("yesssssssssssss")
        self.up_down = up_down(self.tello)
        # self.passing = passWin(self.tello)
        # self.after = after(self.tello)
        # print("outtttttttttttttt")

        self.orient = orient(self.tello)

        self.up = 0
        self.right = 0
        self.trig = 0

    def run(self):

        # while(self.right<2):

            # self.trig, self.up = self.up_down.run()   #up down return two values
            # self.up_down.clear()

            # if(self.trig == 1):
            #     if(self.tello.get_h()>180):
            #         self.up = 1
            #     break

            # else:
        initial_yaw = self.tello.get_yaw()
        while(self.right<2):
            # print("yooooo")
            print("now in up down wali class")
            self.trig = self.up_down.run()
            self.up_down.clear()

            print("now yaw correction started")
            self.orient.orient(initial_yaw)
            # cv2.destroyAllWindows()

            if(self.trig==1):
                if(self.tello.get_h()>180):
                    self.up = 1
                    return self.up,self.right,1
                    
            else:
                print("mow in left right wali class")
                self.right = self.left_right.run(self.right)
                self.left_right.clear()



        return 0,2,0

            # if(self.right>3):


            # print("adgfduasjdjjdahgvafshjskssgvhjxjnvvdh")
            # self.right = self.left_right.run(self.right)
            # self.left_right.clear()

            # if(self.trig == 0):
            #     self.right = self.left_right.run(self.right)   #left right returns only one value
            #     self.left_right.clear()

            # if(self.trig == 1):
            #     self.passing.run()
            #     self.after.run(self.up,self.right)
            #     break

        # self.after.run(self.up,self.right)

        self.tello.land()
        print("Ended")


def main():
    print("now i am gonna start the mission")
    tello = Tello()
    tello.connect()
    tello.streamoff()
    tello.streamon()
    start = starting(tello)
    a,b,c = start.run()
    print("UP is {} and right is {}".format(a,b))

if __name__ == '__main__':
    main()


