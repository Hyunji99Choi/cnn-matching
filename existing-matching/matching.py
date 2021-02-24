import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

imgfile1 = '../df-ms-data/1/df-googleearth-1k-20091227.jpg'
imgfile2 = '../df-ms-data/1/df-googleearth-1k-20181029.jpg'
imgfile1 = '../df-ms-data/1/df-uav-sar-500.jpg'


img1 = cv2.imread(imgfile1,0) # queryImage
img2 = cv2.imread(imgfile2,0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# cnn 기반 매칭 파라미터와 동일
# Flann特征匹配, Flann 특징 일치(FLANN 기반 매칭)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=40)
flann = cv2.FlannBasedMatcher(index_params, search_params)
# 설정한 FLANN으로 매칭
matches = flann.knnMatch(des1, des2, k=2)

'''
# 블로그 매칭 파라미터
# FLANN parameters
FLANN_INDEX_KDTREE = 0

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
'''


# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.3*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()