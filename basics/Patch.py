import cv2
import numpy as np

def find_min_idx(img,patch):
    h,w = np.shape(img)
    ph,pw = np.shape(patch)
    min_x = -1 
    min_y = -1
    error = 1e20
    for i in range(h-ph):
        for j in range(w-pw):
            candidate = img[i:i+ph,j:j+pw]
            e = np.sum(np.sqrt((patch - candidate) * (patch - candidate)))
            if(e<error):
                error = e
                min_y = i 
                min_x = j
    return min_x,min_y  
    
def find_min_idx2(img,patch,center_y,center_x,radius):
    ph,pw = np.shape(patch)
    min_x = -1 
    min_y = -1
    error = 1e20
    for i in range(center_y-radius,center_y+radius):
        for j in range(center_x-radius,center_x+radius):
            candidate = img[i:i+ph,j:j+pw]
            e = np.sum(np.sqrt((patch - candidate) * (patch - candidate)))
            if(e<error):
                error = e
                min_y = i 
                min_x = j
    return min_x,min_y


source = cv2.imread('100.jpg',0)
target = cv2.imread('110.jpg',0)
height,width = np.shape(source)

recons = np.zeros(source.shape,dtype=np.float32)
sumconts = np.ones(source.shape,dtype=np.float32)

ph = 5
pw = 5
radius = 30

for i in range(radius, height - ph-radius,4):
    print(i)
    for j in range(radius, width - pw-radius,4):
        patch = source[i:i+ph,j:j+pw]
        min_x,min_y = find_min_idx2(target,patch,i,j,5)
        nnpatch = target[min_y:min_y+ph,min_x:min_x+pw]
        nnpatch.astype(np.float32)                
        recons[i:i+ph,j:j+pw] += nnpatch
        sumconts[i:i+ph,j:j+pw] += 1.0

recons/=sumconts
recons.astype(np.uint8)
print(np.max(recons))

cv2.imwrite("test.jpg",recons)





