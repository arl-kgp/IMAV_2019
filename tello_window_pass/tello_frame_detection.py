import sys
import cv2
import numpy as np
import math

K1 = 3
split = 5
K2 = 2

font = cv2.FONT_HERSHEY_COMPLEX
maxc = 5
def f1(img):
    img2 = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    
    Z = img2.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500  , 0.01)
    ret,label,center=cv2.kmeans(Z,K1,None,criteria,1,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img2.shape))  
    img = cv2.resize(res2, (img.shape[1], img.shape[0]))

    return img#, center

def f2(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img2 = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

    Z = img2.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500  , 0.01)
    ret,label,center=cv2.kmeans(Z,K2,None,criteria,1,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img2.shape))  
    img = cv2.resize(res2, (img.shape[1], img.shape[0]))

    return img#, center



def splitimg(img):
    
    resimg = np.zeros(img.shape, dtype=np.uint8)
    for i1 in range(split):
        for i2 in range(split):
            sel = np.ix_(np.arange(i1*(img.shape[0]//split), (i1+1)*(img.shape[0]//split)).tolist(), np.arange(i2*(img.shape[1]//split), (i2+1)*(img.shape[1]//split)).tolist(), [0,1,2])
            resimg[(sel)] = f1(img[(sel)])
            #colors.append(c)
            #print(resimg[sel])
            #cv2.imshow('t4', f1(img[(sel)]))
   # print(resimg)
    cv2.imshow('t3', resimg)
    return resimg#, colors

def anglef(a, b, c):
    try:
        aa = abs(math.acos(abs(np.sum((b - a)*(c - b)) / (np.linalg.norm((b - a)) * np.linalg.norm((c - b))))) - math.asin(1))
    except:
        return abs(math.acos(1) - math.asin(1))
    return aa

def get_cnt(img):    

    frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    frame_threshold = cv2.inRange(frame_HSV, (30, 30, 30), (255, 255, 255))/255
    #print(frame_threshold)
    frame_threshold = np.repeat(frame_threshold[:, :, np.newaxis], 3, 2)
    #print(frame_threshold.shape)
    img = np.uint8(frame_threshold*np.int64(img))
    #img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    cv2.imshow("isee", img)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10  , 0.01)
    K = K2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,1,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))  
    #print(center.shape)
    cv2.imshow("isee2", res2)
    cnt3 = []
    cnt2 = []

    maxarea= 0
    for i in range(K):
       # print(np.all(res2==center[i], 2))
        a =  np.uint8(np.all(res2==center[i], 2)*255)
        contours, hierarchy = cv2.findContours(a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnt3.extend(contours)
        for cnt in contours: 
            # Contours detection
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, 0.05*cv2.arcLength(cnt, True), True) # 0.012 param
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            if area > 50:#param

                if len(approx) == 4:

                    #cnt2.append(approx)
                    a1 = anglef(approx[0], approx[1], approx[2]) + anglef(approx[1], approx[2], approx[3]) + anglef(approx[2], approx[3], approx[0]) + anglef(approx[3], approx[0], approx[1])
                    #a1 = 0.9
                    if a1 > 0.8:
                        #cv2.putText(res2, str(((int(a1*100)/100))), (x,y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                        continue
                    
                    if len(cnt) > 4:
                        (cx,cy),(MA,ma),angle = cv2.fitEllipse(cnt)
                        ar = MA/ma
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area)/hull_area
                    else:
                        ar = np.linalg.norm(approx[0] - approx[2])/np.linalg.norm(approx[1]-approx[3])
                        if ar > 1:
                            ar=1/ar
                        solidity = 1.0
                    #cv2.putText(img, str(((int(solidity*100)/100))), (x,y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                    cnt3.append(cnt)
                    #print(solidity)
                    if solidity > 0.9 and maxarea < area and area < img.shape[0]*img.shape[1]*0.5 and area > 800 and ar < 0.5:
                        maxarea = area
                        #print("kya")
                        cnt2.append(approx)
    cv2.imshow("isee3", img)
        
    if len(cnt2)==0:
        return None, cnt3
    #cnt2[-1]*=2
    return cnt2[-1], cnt3

      