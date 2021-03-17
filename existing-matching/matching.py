import cv2
import numpy as np
from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 10

imgfile1 = '../df-ms-data/1/df-googleearth-1k-20091227.jpg'
imgfile2 = '../df-ms-data/1/df-googleearth-1k-20181029.jpg'
imgfile1 = '../df-ms-data/1/df-uav-sar-500.jpg'

#임의로 추가
#imgfile1 = '../df-ms-data/1/df-uav-sar-1k.jpg'
#imgfile2 = '../df-ms-data/1/df-uav-sar-500.jpg'


img1 = cv2.imread(imgfile1,0) # queryImage
img2 = cv2.imread(imgfile2,0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)




'''
# 블로그 매칭 파라미터
# FLANN parameters
FLANN_INDEX_KDTREE = 0

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)


# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# 2순위 매칭 결과의 0.7배보다 더 가까운 값만 취함
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
'''


# cnn 파라미터 -------------

# cnn 기반 매칭 파라미터와 동일
# Flann特征匹配, Flann 특징 일치(FLANN 기반 매칭)
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=40)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 설정한 FLANN으로 매칭
matches = flann.knnMatch(des1, des2, k=2)

goodMatch = []
locations_1_to_use = []
locations_2_to_use = []
# 匹配对筛选 , 매칭 중에서 선택
min_dist = 1000
max_dist = 0
disdif_avg = 0
# 统计平均距离差 , 통계평균거리차
# 매칭 점의 평균 거리차
for m, n in matches:
    disdif_avg += n.distance - m.distance
disdif_avg = disdif_avg / len(matches)
'''
# cnn 방식 그대로 사용
for m, n in matches:
    #自适应阈值 , 자가 적응 임역
    # 좋은 matcing 걸러내기
    if n.distance > m.distance + disdif_avg:
        goodMatch.append(m)
'''
# cnn 방식으로 수정
matchesMask = [[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if n.distance > m.distance + disdif_avg:
        matchesMask[i]=[1,0]

#-----------

# 기존
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)


img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()