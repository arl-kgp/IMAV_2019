#!/usr/bin/env python3
import sys
import os.path
import orbslam2
import time
import cv2
import numpy as np
import asyncio

from djitellopy import Tello
import imutils as im
from time import sleep

from threading import Thread

fc = 0
cfc = 0
buffer_frame = np.zeros((640, 720))

cp = False

def imageCallback():
    tello = Tello()
    global buffer_frame, fc
    tello.connect()
    tello.streamoff()
    tello.streamon()

    capture = tello.get_video_capture()

    while 1:
        #print("SSSSSSSSSSSSSSSSSSSSSSSS")
        ret, frame = capture.read()
        frameBGR = np.copy(frame)
        if not cp:
            buffer_frame = im.resize(frameBGR,width=720)
        #cv2.imshow('win', buffer_frame)
        #cv2.waitKey(1)
        fc+=1
        #print(fc)
        
def get_new_frame():
    global buffer_frame, fc, cfc
    while (fc == cfc or fc <= 5):
        sleep(0.01)
        #print(fc)
    cfc = fc
    cp = True
    rv = np.copy(buffer_frame)
    cp = False
    return rv



def main(settings_path):

#    image_filenames, timestamps = load_images(sequence_path)
#    num_images = len(image_filenames)
    p = Thread(target=imageCallback)
    # you have to set daemon true to not have to wait for the process to join
    #p.daemon = True
    p.start()
    #sleep(5)
    slam = orbslam2.System('ORBvoc.txt', settings_path, orbslam2.Sensor.MONOCULAR)
    slam.set_use_viewer(True)
    slam.initialize()

    print('-----')
    print('Start processing sequence ...')
    times_track = []
    
    p = [0,0,0]
    r = None
    while 1:
        frame = get_new_frame()
        #print("sDDDDDDDDDDDDDDDDDDDDd")
        # Display the resulting frame
        #cv2.imshow('Frame',frame)
    
        # Press Q on keyboard to  exit
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    print("2")
        #    break
        image = frame
        #tframe = cap.get(cv2.CAP_PROP_POS_MSEC)
        

        t1 = time.time()
        pose = slam.process_image_mono(image, 0)
        if  not (pose is None):
            R = pose[:3,:3]
            T = pose[[0,1,2],[3]].reshape(3,1)
            print(R.shape)
            print(T.shape)
            print(-np.matmul(R.T, T))
            #p2(pose[[0,1,2,3],[3]])
        #print(slam.process_image_mono(image, tframe))
        #slam.
        
        t2 = time.time()

        ttrack = t2 - t1
        times_track.append(ttrack)
    
    # Closes all the frames
    cv2.destroyAllWindows()    

    save_trajectory(slam.get_trajectory_points(), 'trajectory.txt')

    slam.shutdown()
    p.join()
    times_track = sorted(times_track)
    total_time = sum(times_track)
    print('-----')
    print('median tracking time: {0}'.format(times_track[len(times_track) // 2]))
    print('mean tracking time: {0}'.format(total_time / len(times_track)))

    sleep(100)
    return 0


def save_trajectory(trajectory, filename):
    with open(filename, 'w') as traj_file:
        traj_file.writelines('{time} {r00} {r01} {r02} {t0} {r10} {r11} {r12} {t1} {r20} {r21} {r22} {t2}\n'.format(
            time=repr(t),
            r00=repr(r00),
            r01=repr(r01),
            r02=repr(r02),
            t0=repr(t0),
            r10=repr(r10),
            r11=repr(r11),
            r12=repr(r12),
            t1=repr(t1),
            r20=repr(r20),
            r21=repr(r21),
            r22=repr(r22),
            t2=repr(t2)
        ) for t, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 in trajectory)


if __name__ == '__main__':
    main(sys.argv[1])
