import numpy as np
import cv2

# img = cv2.imread('100.jpg')
# fast=cv2.FastFeatureDetector_create(threshold=20,nonmaxSuppression=True,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
# kp = fast.detect(img,None)
# img = cv2.drawKeypoints(img,kp,img,color=(255,0,0))
# cv2.imshow('',img)
# cv2.waitKey(0)

frame1 = cv2.imread('100.jpg')
frame2 = cv2.imread('110.jpg')

# prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# hsv[...,1] = 255

# next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
# flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
# hsv[...,0] = ang*180/np.pi/2
# hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
# rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

# cv2.imwrite('opticalfb.png',frame2)
# cv2.imwrite('opticalhsv.png',rgb)

# # params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
old   = cv2.imread('100.jpg')
frame = cv2.imread('110.jpg')
old_gray = cv2.cvtColor(old,cv2.COLOR_BGR2GRAY)
frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)



good_new = p1[st==1]
good_old = p0[st==1]

print(good_new.shape)


for pt in good_new:
    cv2.circle(old,(pt[0],pt[1]),2,[255,255,0],-1)

for pt1 in good_old:
    cv2.circle(frame,(pt1[0],pt1[1]),2,[255,255,0],-1)

cv2.imwrite("s.jpg",old)
cv2.imwrite("t.jpg",frame)