import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('stepler//orig.png',0)          # queryImage
img2 = cv2.imread('stepler//(33).jpg',0) # trainImage
#img1 = cv2.resize(img1, (600, 800))
#img2 = cv2.resize(img2, (600, 800))
plt.imshow(img2),plt.show()
# Initiate SIFT detector
orb=cv2.SIFT_create()
#orb=cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
#print(des1[0])
print( des2)
img5 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
plt.imshow(img5,),plt.show()
img6 = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
plt.imshow(img6,),plt.show()
# FLANN parameters
#FLANN_INDEX_LSH= 6
#index_params= dict(algorithm = FLANN_INDEX_LSH,table_number = 6,key_size = 12,multi_probe_level = 1)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i in range(len(matches)):
    if( len(matches[i])>1 ):
        if matches[i][0].distance < 0.7*matches[i][1].distance:
            matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()
