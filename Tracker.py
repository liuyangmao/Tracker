import numpy as np
import cv2

VARIANCE = 4
MAXTRACKS = 800
GRIDSIZE = 15
MASK_RADIUS = 15

class GridTracker:
  
  #mask used for ignoring regions of the image in the detector and for maintaining minimal feature distance
  curMask = 0
  
  #tracked feas of current frame
  trackedFeas = [] #matched Feas of currFrame

  #all feas of current frame
  allFeas = []
  preFeas = [] #matched Feas of preFrame

  rows = 0
  cols = 0
  
  #num of feas from last frame
  numActiveTracks = 0
  TRACKING_HSIZE = LK_PYRAMID_LEVEL = MAX_ITER = fealimitGrid = 0
  ACCURACY = LAMBDA = 0
  usableFrac = 0

  #store image pyramid for re-utilizatio
  prevPyr = 0
  prevIm = 0
  overflow = MaxTracks = 0

  #grids devision
  hgrids_x = 0
  hgrids_y = 0
  hgrids_total = 0
  #records feature number of each grids
  feanumofGrid = []

  minAddFrac = minToAdd = 0.0
  unusedRoom = gridsHungry = 0
  lastNumDetectedGridFeatures = []
  hthresholds = []
  DETECT_GAIN = 0

  detector = []

  def __init__(self):
    return

  def maskPoint(self,y,x):
    if self.curMask[y,x] == 0:
      return 1
    xx = np.int(x - MASK_RADIUS / 2 + .5)
    yy = np.int(y - MASK_RADIUS / 2 + .5)
    w  = np.int(x + MASK_RADIUS / 2 + .5)
    h  = np.int(y + MASK_RADIUS / 2 + .5)
    cv2.rectangle(self.curMask,(xx,yy),(w,h),0, -1)
    return 0

  def ParameterInit(self,im):
    self.rows, self.cols = im.shape
    self.numActiveTracks = 0
    self.TRACKING_HSIZE = 8
    self.LK_PYRAMID_LEVEL = 4
    self.MAX_ITER = 10
    self.ACCURACY = 0.1
    self.LAMBDA = 0.0

    self.hgrids_x = GRIDSIZE
    self.hgrids_y = GRIDSIZE
    self.hgrids_total = self.hgrids_x * self.hgrids_y

    self.usableFrac = 0.02
    self.MaxTracks = MAXTRACKS
    self.minAddFrac = 0.1
    self.minToAdd = self.minAddFrac * self.MaxTracks

    self.fealimitGrid = np.floor(self.MaxTracks / (self.hgrids_total))
    self.lastNumDetectedGridFeatures = [0] * self.hgrids_total
    
    self.DETECT_GAIN = 10
    self.hthresholds = [20] * self.hgrids_total
    self.feanumofGrid = [0] * self.hgrids_total

  def SpaceInit(self,im):
    self.curMask = np.ones( (self.rows, self.cols), dtype = np.uint8)
    for i in range(self.hgrids_total):
      fast = cv2.FastFeatureDetector_create(self.hthresholds[i],nonmaxSuppression=True,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)  
      self.detector.append(fast)
    self.prevIm = im
    
  def Calc_optical_flow_if_feas_from_last_frame(self, im1):
    
    lk_params = dict( winSize  = (2 * self.TRACKING_HSIZE + 1, 2 * self.TRACKING_HSIZE + 1),maxLevel = self.LK_PYRAMID_LEVEL,
                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.MAX_ITER, self.ACCURACY))
    
    #perform LK tracking from OpenCV, parameters matter a lot
    self.preFeas = []
    self.trackedFeas = []
    
    p0 = np.asarray(self.allFeas)
    
    p1, st, error = cv2.calcOpticalFlowPyrLK(self.prevIm, im1, p0, None, **lk_params)
    
    self.prevIm = im1
    
    #clear feature counting for each grid
    self.feanumofGrid = [0] * self.hgrids_total
    
    for i in range(len(p1)):
      if st[i] and p1[i,0] > self.usableFrac * self.cols \
      and p1[i,0] < (1.0 - self.usableFrac) * self.cols  \
      and p1[i,1] > self.usableFrac * self.rows  \
      and p1[i,1] < (1.0 - self.usableFrac) * self.rows:
          
        shouldKill = self.maskPoint(np.int(p1[i,1]), np.int(p1[i,0]))
        
        if shouldKill:
          self.numActiveTracks -= 1
        else:
          self.preFeas.append(p0[i])
          self.trackedFeas.append(p1[i])

          self.hgridIdx = np.int(np.floor(p1[i,0] / (self.cols / self.hgrids_x)) \
          + self.hgrids_x * np.floor(p1[i,1] / (self.rows / self.hgrids_y)))

          self.feanumofGrid[self.hgridIdx] += 1
      else:
        self.numActiveTracks -= 1
    
  def Sampling(self,im1):
    #unusedRoom sum
      unusedRoom = 0

      #hungry grids
      hungryGrid_idx = []
      hungryGrid_value = []

      #room for adding featurs to each grid
      room = 0

      #the hungry degree of a whole frame 
      hungry = 0

      #keypoints detected from each grids
      sub_keypoints = [0] * self.hgrids_total
      sub_keypoints_list = [0] * self.hgrids_total

      midGrid = np.floor((self.hgrids_total - 1) / 2.0)
      
      for q in range(self.hgrids_total):
        if self.numActiveTracks < self.MaxTracks:
          i = np.int(q)
          if q == 0:
            i = np.int(midGrid)
          if q == midGrid:
            i = 0  

          room = self.fealimitGrid - self.feanumofGrid[i]
          
          if room > self.fealimitGrid * self.minAddFrac:
            
            celly = np.int( i / self.hgrids_x )
            cellx = np.int(i - celly * self.hgrids_x)
            row_start = np.int((celly * self.rows) / self.hgrids_y)
            row_size = np.int(((celly + 1) * self.rows) / self.hgrids_y - row_start)
            col_start = np.int((cellx * self.cols) / self.hgrids_x)
            col_size = np.int(((cellx + 1) * self.cols) / self.hgrids_x - col_start)
            
            sub_image = im1[row_start:row_start+row_size,col_start:col_start+col_size]
            sub_mask = self.curMask[row_start:row_start+row_size,col_start:col_start+col_size]
            
            lastP = (self.lastNumDetectedGridFeatures[i] - 15.0 * room) / (15.0 * room)
            newThresh = self.detector[i].getThreshold()
            newThresh = newThresh + np.ceil(self.DETECT_GAIN * lastP)
            if newThresh > 200:
              newThresh = 200
            if newThresh < 5:
              newThresh = 5
            
            self.detector[i].setThreshold(np.int(newThresh))
            sub_keypoints[i] = self.detector[i].detect(sub_image,sub_mask)            
            self.lastNumDetectedGridFeatures[i] = len(sub_keypoints[i])
            
            sub_keypoints_list[i] = np.float32([it.pt for it in sub_keypoints[i]])
            
            n = 0
            j = 0

            #first round
            while(1):
              if j<len(sub_keypoints[i]) and n<room and self.numActiveTracks < self.MaxTracks:
                pass
              else:
                break
              
              pt = np.copy(sub_keypoints_list[i][j])
              pt[0] += col_start
              pt[1] += row_start
              
              ptx = np.round(pt[0])
              pty = np.round(pt[1])

              if self.curMask[np.int(pty),np.int(ptx)] != 0:
                u = np.int(ptx-MASK_RADIUS/2+.5)
                v = np.int(pty-MASK_RADIUS/2+.5)
                w = np.int(ptx+MASK_RADIUS/2+.5)
                z = np.int(pty+MASK_RADIUS/2+.5)
                cv2.rectangle(self.curMask,(u,v),(w,z),0, -1)
                self.allFeas.append(pt)
                self.numActiveTracks += 1
                n+=1
              j+=1

            if n <= room:
              hungryGrid_idx.append(i)
              hungryGrid_value.append(len(sub_keypoints[i]) - j)
              hungry += hungryGrid_value[-1]
      
      #begin of second round
      unusedRoom = self.MaxTracks - self.numActiveTracks

      if unusedRoom > self.minToAdd:
        for i in range(len(hungryGrid_idx)):
          first = np.int(hungryGrid_idx[i])
          second = hungryGrid_value[i]

          if first >= len(hungryGrid_value):
            continue

          celly = np.int( first / self.hgrids_x )
          cellx = np.int(first - celly * self.hgrids_x)
          row_start = np.int((celly * self.rows) / self.hgrids_y)
          col_start = np.int((cellx * self.cols) / self.hgrids_x)
          
          room = (unusedRoom * hungryGrid_value[first]) / hungry
          m = 0
          j = 0
          while(1):
            if m<room and j<(len(sub_keypoints[first])- second):
              pass
            else:
              break
            
            pt = np.copy(sub_keypoints_list[first][j])

            pt[0] += col_start
            pt[1] += row_start
            ptx = np.round(pt[0])
            pty = np.round(pt[1])
            if self.curMask[np.int(pty),np.int(ptx)] != 0:
                u = np.int(ptx-MASK_RADIUS/2+.5)
                v = np.int(pty-MASK_RADIUS/2+.5)
                w = np.int(ptx+MASK_RADIUS/2+.5)
                z = np.int(pty+MASK_RADIUS/2+.5)
                cv2.rectangle(self.curMask,(u,v),(w,z),0, -1)
                self.allFeas.append(pt)
                m += 1
            j+=1

  def trackerInit(self,im):
    self.ParameterInit(im)
    self.SpaceInit(im)
    self.Update(im)
  
  def Update(self,im1):
    self.curMask[:,:] = 1
    self.numActiveTracks = len(self.allFeas)
    
    #do optical flow if there are feas from last frame
    if self.numActiveTracks > 0:
      self.Calc_optical_flow_if_feas_from_last_frame(im1)
    else:
      self.feanumofGrid = [0] * self.hgrids_total 
      
    #self.allFeas = self.trackedFeas
    self.allFeas = []
    self.allFeas = self.trackedFeas.copy()
    
    ntoadd = self.MaxTracks - self.numActiveTracks
    
    if ntoadd > self.minToAdd:
      self.Sampling(im1)
    
if __name__ == '__main__':
  frames = []
  grays = []
  numFrames = 200

  for i in range(numFrames):
    name = "./frames/%04d.png" % i 
    frame = cv2.imread(name)
    frames.append(frame)
    gray = cv2.imread(name,0)
    grays.append(gray)
  print("end loading frame\n")
  
  gt = GridTracker()
  gt.trackerInit(grays[0])

  for i in range(1,numFrames):
    gt.Update(grays[i])  
    for j in range(len(gt.preFeas)):
      pt0 = gt.preFeas[j]
      cv2.circle(frames[i-1],(pt0[0],pt0[1]),2,(0,255,0),-1)
    name = "./debug/%04d.jpg" % (i-1)
    cv2.imwrite(name,frames[i-1])
  
# img = cv2.imread('100.jpg')
# fast=cv2.FastFeatureDetector_create(threshold=20,nonmaxSuppression=True,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
# kp = fast.detect(img,None)
# img = cv2.drawKeypoints(img,kp,img,color=(255,0,0))
# cv2.imshow('',img)
# cv2.waitKey(0)

# frame1 = cv2.imread('100.jpg')
# frame2 = cv2.imread('110.jpg')

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
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )

# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# old   = cv2.imread('100.jpg')
# frame = cv2.imread('110.jpg')
# old_gray = cv2.cvtColor(old,cv2.COLOR_BGR2GRAY)
# frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
# good_new = p1[st==1]
# good_old = p0[st==1]

# for pt in good_new:
#     cv2.circle(old,(pt[0],pt[1]),2,[255,255,0],-1)

# for pt1 in good_old:
#     cv2.circle(frame,(pt1[0],pt1[1]),2,[255,255,0],-1)

# cv2.imwrite("s.jpg",old)
# cv2.imwrite("t.jpg",frame)