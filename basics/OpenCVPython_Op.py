import numpy as np
import cv2

# #resize
# img = cv2.imread('100.jpg')
# height,width = img.shape[:2]
# res = cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
# cv2.imshow("",res)
# cv2.waitKey(0)

# #rotation
# img = cv2.imread('100.jpg')
# rows,cols = img.shape[:2]
# M = cv2.getRotationMatrix2D((cols/2,rows/2),-10,1)
# res = cv2.warpAffine(img,M,(cols,rows))
# cv2.imshow("",res)
# cv2.waitKey(0)

def feature_matches(img1,img2):
    s = cv2.xfeatures2d.SURF_create(400)
    kp1, des1 = s.detectAndCompute(img1, None)
    kp2, des2 = s.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k = 2)
    goodMatch = []
    for m,n in matches:
        if m.distance < 0.50*n.distance:
            goodMatch.append(m)
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    post1 = np.float32([kp1[pp].pt for pp in p1])
    post2 = np.float32([kp2[pp].pt for pp in p2])
    return post1, post2

img1 = cv2.imread("100.jpg")
img2 = cv2.imread("110.jpg")

img1_fea, img2_fea = feature_matches(img1,img2)

H, mask = cv2.findHomography(img1_fea, img2_fea, cv2.RANSAC,3.0)
matchesMask = mask.ravel().tolist()
rows,cols = img1.shape[:2]
res = cv2.warpPerspective(img1,H,(cols,rows))
cv2.imwrite("sw.jpg",res)

# for i in range(len(img1_fea)):
#     cv2.circle(img1,(img1_fea[i][0],img1_fea[i][1]),2,[255,255,0],-1)
#     cv2.circle(img2,(img2_fea[i][0],img2_fea[i][1]),2,[255,255,0],-1)
# cv2.imwrite('s.jpg',img1)
# cv2.imwrite('t.jpg',img2)

# def drawMatchesKnn_cv2(img1,kp1,img2,kp2,goodMatch):
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]
 
#     vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
#     vis[:h1, :w1] = img1
#     vis[:h2, w1:w1 + w2] = img2
 
#     p1 = [kpp.queryIdx for kpp in goodMatch]
#     p2 = [kpp.trainIdx for kpp in goodMatch]
 
#     post1 = np.int32([kp1[pp].pt for pp in p1])
#     post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
 
#     for (x1, y1), (x2, y2) in zip(post1, post2):
#         cv2.line(vis, (x1, y1), (x2, y2), (0,0,255))
 
#     cv2.namedWindow("match")
#     cv2.imshow("match", vis)

# img1 = cv2.imread("100.jpg")
# img2 = cv2.imread("110.jpg")
# s = cv2.xfeatures2d.SIFT_create()
# #s = cv2.xfeatures2d.SURF_create(400)
# kp1, des1 = s.detectAndCompute(img1, None)
# kp2, des2 = s.detectAndCompute(img2, None)
# bf = cv2.BFMatcher(cv2.NORM_L2)
# matches = bf.knnMatch(des1, des2, k = 2)
# goodMatch = []
# for m,n in matches:
#     if m.distance < 0.50*n.distance:
#         goodMatch.append(m)
# drawMatchesKnn_cv2(img1,kp1,img2,kp2,goodMatch)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## Video read
# cap = cv2.VideoCapture("./0002.avi")
# ret = True
# c=0
# while(ret):
# 	ret, frame = cap.read()
# 	cv2.imshow('capture',frame)
# 	cv2.waitKey(20)
# 	c+=1
# 	if c==110:
# 		cv2.imwrite('110.jpg',frame)
# 		break

## Video capture
# cap = cv2.VideoCapture(0)
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()# cv2.destroyAllWindows()

## Video writer
# fourcc = cv2.VideoWriter_fourcc(*'x264') #XVID, x264
# out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (640,480))
# cap = cv2.VideoCapture(0)
# while(True):
#     ret, frame = cap.read()
#     cv2.imshow('frame',frame)
#     frame = cv2.flip(frame,1)
#     out.write(frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
# out.release()

## Draw line functions
# image = np.zeros((360,640,3),dtype=np.uint8)
# start_point = (10,10)
# end_point = (100,100)
# color = (0,255,0)
# line_width = 4
# line_type = 8
# cv2.line(image, start_point, end_point, color, line_width, line_type)
# cv2.namedWindow("")
# cv2.imshow("",image)
# cv2.waitKey(0)

## Draw point function
# image = np.zeros((360,640,3),dtype=np.uint8)
# xs = []
# ys = []
# for i in range(50):
#     x,y = np.random.randint(0,640),np.random.randint(0,360)
#     xs.append(x)
#     ys.append(y)
# for i in range(50):
#     cv2.circle(image,(xs[i],ys[i]),5,(55,255,155),-1)
# cv2.namedWindow("")
# cv2.imshow("",image)
# cv2.waitKey(0)

## copy to
# I1 = cv2.imread('100.jpg')
# I2 = cv2.imread('110.jpg')
# h,w,c = np.shape(I1)
# #print( str(w) +' ' + str(h) +' '+str(c))
# image = np.zeros((h,w*2,3),I1.dtype)
# image[0:h,0:w,:] = I1
# image[0:h,w:2*w,:] = I2
# cv2.namedWindow("")
# cv2.imshow("",image)
# cv2.waitKey(0)

## split & merge
# img = cv2.imread('100.jpg')
# b,g,r = cv2.split(img)
# #cv2.imshow('Blue',b)
# #cv2.imshow('Green',g)
# #cv2.imshow('Red',r)
# merged = cv2.merge([b,g,r])
# cv2.imshow("",merged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #put text
# img = np.zeros((360,640,3),dtype=np.uint8)
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,'OpenCV',(10,200), font, 4,(255,255,255),2,cv2.LINE_AA)
# cv2.imshow("",img)
# cv2.waitKey(0)

# fast feature
# img = cv2.imread('100.jpg')
# fast=cv2.FastFeatureDetector_create(threshold=20,nonmaxSuppression=True,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)#获取FAST角点探测器
# kp=fast.detect(img,None)#描述符
# img = cv2.drawKeypoints(img,kp,img,color=(255,0,0))#画到img上面
# for it in kp:
#     it.pt[0] = 2
#     print(str(it.pt[0])+ " " +str(it.pt[0]))
#cv2.imshow('',img)
#cv2.waitKey(0)

## goodFeaturesToTrack
# img = cv2.imread('100.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
# corners = np.int0(corners)
# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img,(x,y),3,255,-1)
# cv2.imshow('',img)
# cv2.waitKey(0)

## surf feature
# img = cv2.imread('100.jpg',0)
# #s = cv2.xfeatures2d.SURF_create(400)
# s = cv2.xfeatures2d.SIFT_create()
# key_query,desc_query = s.detectAndCompute(img,None)
# img=cv2.drawKeypoints(img,key_query,img)
# cv2.imshow('sp',img)
# cv2.waitKey(0)









