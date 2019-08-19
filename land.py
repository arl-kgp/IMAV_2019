from djitellopy import Tello
import time


tello = Tello()
tello.connect()
time.sleep(5)

tello.land()
time.sleep(2)

tello.end()