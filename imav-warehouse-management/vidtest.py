from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import re
from djitellopy import Tello
import imutils as imu
from time import sleep  
import time   
import os
import random
import time
import asyncio
import itertools
import numpy as np
import cv2
import PIL.Image as pim
import pytesseract
from qrcode import *
from text import *
from imutils.object_detection import non_max_suppression
from PIL import Image
import scipy
import scipy.misc
from csvParserfinal import *
import final_csv_final
import threading

from JoyStick_Controller.controller_module import Controller
import JoyStick_Controller.xbox as xbox


import math

K1 = 3
split = 5
K2 = 6

font = cv2.FONT_HERSHEY_COMPLEX
maxc = 5

id = 1

def order_points(pts):

    pts = pts.reshape(4,2)
    rect = np.zeros((4, 2), dtype = "float32")
    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # return the ordered coordinates
    return rect

def get_cnt(frame):
    arSet = 0.5
    kernel = np.ones((5,5),np.uint8)#param 1

    blurred = cv2.GaussianBlur(frame, (7, 7), 0)#param 1

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    dilS = cv2.dilate(s,kernel,iterations = 1)
    newS = dilS-s
    newS = cv2.equalizeHist(newS)
    # newS = cv2.GaussianBlur(newS, (11, 11), 0)


    dilV = cv2.dilate(v,kernel,iterations = 1)#param 1
    newV = dilV-v
    newV = cv2.equalizeHist(newV)

    dilH = cv2.dilate(h,kernel,iterations = 1)
    newH = dilH-h
    newH = cv2.equalizeHist(newH)


    sabKaAnd = cv2.bitwise_or(newS,newV)
    kernel2 = np.ones((3,3),np.uint8)#param 1
    sabKaAnd = cv2.erode(sabKaAnd,kernel2,iterations = 1)#param 1
    sabKaAnd = cv2.erode(sabKaAnd,kernel2,iterations = 1)#param 1

    sabKaAnd = cv2.dilate(sabKaAnd,kernel2,iterations = 1)#param 1
    sabKaAnd = cv2.GaussianBlur(sabKaAnd, (11, 11), 0)

    maskSab = cv2.inRange(sabKaAnd,120,255)#param 1****

    maskSab = cv2.erode(maskSab,kernel2,iterations = 1)
    maskSab = cv2.dilate(maskSab,kernel2,iterations = 1)

    maskSab = cv2.bitwise_and(maskSab,newV)
    maskSab = cv2.equalizeHist(maskSab)
    maskSab = cv2.inRange(maskSab,190,255)# param *****

    kernel2 = np.ones((2,2),np.uint8) #param ****
    maskSab = cv2.erode(maskSab,kernel2,iterations = 1)
    maskSab = cv2.dilate(maskSab,kernel2,iterations = 1)
    mask = maskSab
    cv2.imshow("msk", mask)
    # Contours detection
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    oldArea = 300
    cnt2 = []
    lcnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.012*cv2.arcLength(cnt, True), True) # 0.012 param
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        

        if area > oldArea:#param
            # if len(approx) == 3:
                # cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))
            if len(approx) == 4:
                
                if 0:
                    (cx,cy),(MA,ma),angle = cv2.fitEllipse(cnt)
                    ar = MA/ma
                else:
                    ar = (np.linalg.norm(approx[0] - approx[1]) + np.linalg.norm(approx[2] - approx[3]))/(np.linalg.norm(approx[2]-approx[1])+np.linalg.norm(approx[0]-approx[3]))
                    if ar > 1:
                        ar=1/ar

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area)/hull_area

                condition = ar < 0.4 and ar > 0.28
                if solidity > 0.95 and condition:
                    cnt2.append(approx)
                    oldArea = area
                    lcnt = approx

        
    if len(cnt2)==0:
        return None, cnt2
    #cnt2[-1]*=2
    cnt1 = order_points(lcnt)
    cy = np.sum(cnt1[:, 1])/4
    cx = np.sum(cnt1[:, 0])/4

    param1 = 0.65
    param2 = 0.45

    cnt1[0][1] = cy - (cy-cnt1[0][1] )*param1
    cnt1[1][1] = cy - (cy-cnt1[1][1] )*param1
    cnt1[2][1] = cy + (-cy+cnt1[2][1] )*param1
    cnt1[3][1] = cy + (-cy+cnt1[3][1] )*param1
    cnt1[0][0] = cx - (cx-cnt1[0][0] )*param2
    cnt1[1][0] = cx - (cx-cnt1[1][0] )*param2
    cnt1[2][0] = cx + (-cx+cnt1[2][0] )*param2
    cnt1[3][0] = cx + (-cx+cnt1[3][0] )*param2
    
    c = np.reshape(cnt1, (4,1,2))
    rect = cv2.boundingRect(c)

    x,y,w,h = rect
    cropped = frame[y: y+h, x: x+w]

    return c, cropped

def text_better(text):
    list1 = list(text)
    if len(text) == 4:
        list1 = list1[1:]

    if(text[0]=='I'):
        list1[0]='1'
    elif(text[0]=='A'):
        list1[0]='4'

    if(text[1]=='I'):
        list1[1]='1'
    elif(text[1]=='A'):
        list1[1]='4'

    if(text[2]=='4'):
        list1[2]='A'
    elif(text[2]=='3'):
        list1[2]='B'
    elif(text[2]=='8'):
        list1[2]='B'

    text = ''.join(list1)
    return text

def roi_detect(image):
        # initialize the list of results
    results = []
    text_list = []
    conf_list = []
    corners = []

    output = image.copy()

    contour, roi = get_cnt(image)

    if(contour is not None):
        cv2.imshow("roi",roi)
        im, text, conf = return_text(roi) 

        x,y,w,h = cv2.boundingRect(contour)

        text = text.replace('_', '')
        text = text.replace('\\', '')
        text = text.replace('/', '')
        
        if len(text) == 3 or len(text) == 4:
            #text = text_better(text)
            text_list.append(text)
            conf_list.append(conf)
            corners.append((x,y,x+w,y+h))

        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(output, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return text_list, conf_list, corners, output

def decode_predictions(scores, geometry, min_confidence):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)



def apply_contrast(im):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(im_hsv)
    ret, v = cv2.threshold(v,127,255,cv2.THRESH_BINARY)
    im_hsv[:, :, 2] = v
    im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
    return im

def apply_thresh(img):
    lower = np.array([50, 50, 50])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower, upper)
    img = cv2.bitwise_and(img, img, mask = mask)


def img_resize(im):
    fx = 7.092159469231584126e+02
    fy = 7.102890453175559742e+02
    cx = 3.681653710406367850e+02
    cy = 2.497677007139825491e+02

    """fx = 672.074266
    fy = 672.019640
    cx = 324.846853
    cy = 255.070573"""

    depth = 200
    real_text_w = 150    #200
    real_text_h = 60    #100
    favg = (fx+fy)/2
    text_w = (real_text_w*favg)/depth
    text_h = (real_text_h*favg)/depth

    optical_text_w = 172
    optimal_text_h = 74
    k = optimal_text_h/text_h
    rows = int(im.shape[0] * 1.2)
    cols = int(im.shape[1] * 1.2)
    dim = (cols, rows)
    resized = cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR)
    return resized


def undistort(im):
    K = np.array([[5.277994366999020031e+02,0.000000000000000000e+00,3.711893449350251331e+02], [0.000000000000000000e+00,5.249025134499009937e+02,2.671209192674019732e+02], [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]], dtype = 'uint8')

    dist = np.array([-1.941494206892808161e-01,-1.887639714668869206e-02,-5.988986169837847741e-03,-7.372353351255917582e-05,7.269696522356267065e-02], dtype = 'uint8')

    K_inv = np.linalg.inv(K)

    h , w = im.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

    mapx,mapy = cv2.initUndistortRectifyMap(K,dist,None,newcameramtx,(w,h),5)
    dst = cv2.remap(im,mapx,mapy,cv2.INTER_LINEAR)

    x,y,w,h = roi
    im = dst[y:y+h,x:x+w]

    #print("ROI: ",x,y,w,h)
    #cv2.imshow("lkgs",frame2use)

    return(im)

def hist_equalise(im):

    im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    im_yuv[:,:,0] = cv2.equalizeHist(im_yuv[:,:,0])

    # convert the YUV image back to RGB format
    im = cv2.cvtColor(im_yuv, cv2.COLOR_YUV2BGR)
    return(im)

def text_finder(im):

    # EAST + Tesseract
    text = None
    text_list, conf_list, corners, output = roi_detect(im)
    # if(corners):
    #     print("Area: "+str(find_area(corners)))

    text_list_ref = []            ## FINAL RETURN VALUES
    conf_list_ref = []
    corners_ref = []

    for index in range(len(text_list)):
        text = text_list[index]
        if check_format(text):
            text_list_ref.append(text_list[index])
            conf_list_ref.append(conf_list[index])
            corners_ref.append(corners[index])
            print("Added: "+str(text))
    if len(conf_list_ref) > 0:   # Use in left
        text = text_list_ref[np.argmax(conf_list_ref)]
        corner_pts = corners_ref[np.argmax(conf_list_ref)]
    
    return text, corners, output

def check_format(text):
    to_print = 0
    rex1 = re.compile("^[0-9]{2}[A-Z]$")
    rex2 = re.compile("^[0-9][A-Z]$")
    text = str(text)
    if rex1.match(text) or rex2.match(text):
        to_print = 1

    return to_print

def write_in_file(qrlist, text):

    for i in range(len(qrlist)):
        Data = qrlist[i]
        if(Data):
            Data = str(Data).strip("b'")
        # f = open('warehouse.csv','a')
        f.write('%s,%s,\n'%(Data, text))

def find_text_and_write(im, qrlist):
    text, corners, output = text_finder(im)
    print(text)
    
    check = check_format(text)

    if text != None and corners:
        write_in_file(qrlist, text)

    return output

        
if __name__ == '__main__':

    f = open('warehouse.csv','w')
    f1 = open('out2.csv', 'w')
    cap = cv2.VideoCapture('3.mp4')
    while True:
        
        rcOut = np.zeros(4)

        # for FPS:
        start_time = time.time()
        
        # Frame preprocess
        _, frameBGR = cap.read()

        #im = imu.resize(frameBGR, width=720)

        # Undistortion --Uncomment for Tello-001
        #im = undistort(im)
        im = frameBGR
        # QR-codes detect
        if im is None:
            break
        im, qrpoints, qrlist = main(im)

        if qrpoints != []:            # If QR detected, detect TEXT
            print(qrlist)
            for subset in itertools.combinations(qrlist, 2):
                d1, d2 = subset[0], subset[1]
                if d1 is not None and d2 is not None:
                    d1 = str(d1).strip("b'")
                    d2 = str(d2).strip("b'")
                    f1.write('%s,%s,\n'%(d1, d2))

            #im = img_resize(im)
            #im = apply_contrast(im)
            #im = apply_thresh(im)
            #im = hist_equalise(im)

            im = find_text_and_write(im, qrlist)
                

        cv2.imshow("Results", im)
        cv2.waitKey(1)
        

        #print self.rcOut
        rcOut = [0,0,0,0]
    try:
        f.close()
    except:
        pass
    try:
        f1.close()
    except:
        pass
    sleep(5)
    parser1()
    sleep(5)
    final_csv_final.getout(5, "distribution.csv")

    




      
