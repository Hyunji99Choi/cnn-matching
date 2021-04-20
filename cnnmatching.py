import argparse
import cv2
import numpy as np
import imageio
import plotmatch
from lib.cnn_feature import cnn_feature_extract
import matplotlib.pyplot as plt
import time
from skimage import measure
from skimage import transform

#time count
start = time.perf_counter()

_RESIDUAL_THRESHOLD = 30
#Test1nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8
imgfile1 = 'df-ms-data/1/df-googleearth-1k-20091227.jpg'
imgfile2 = 'df-ms-data/1/df-googleearth-1k-20181029.jpg'
imgfile1 = 'df-ms-data/1/df-uav-sar-500.jpg'


# 실험을 위해 임의로 추가
imgfile1 = 'df-ms-data/same_img/view1.png'
imgfile2 = 'df-ms-data/same_img/view5.png'

## 비교 실험 논문
imgfile1 = 'df-ms-data/same_img/view1_5.png'
imgfile2 = 'df-ms-data/same_img/view5_5.png'


start = time.perf_counter()

# read left image
image1 = imageio.imread(imgfile1)
image2 = imageio.imread(imgfile2)

# 이미지를 읽어오는데 걸린 시간
print('read image time is %6.3f' % (time.perf_counter() - start))

start0 = time.perf_counter()

# lib - cnn_feature.py
# cnn을 이용하여 keypoint, 특징점 점수, 기술자 검출 (이미지에서 특징점 검출)
# 기능 추출을 D2-Net 사용하는 것 같음..
kps_left, sco_left, des_left = cnn_feature_extract(image1,  nfeatures = -1)
kps_right, sco_right, des_right = cnn_feature_extract(image2,  nfeatures = -1)

# cnn으로 특징 추출하는데 걸린 시간
print('Feature_extract time is %6.3f, left: %6.3f,right %6.3f' % ((time.perf_counter() - start), len(kps_left), len(kps_right)))
start = time.perf_counter()

# Flann特征匹配, Flann 특징 일치(FLANN 기반 매칭)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=40)
flann = cv2.FlannBasedMatcher(index_params, search_params)
# 설정한 FLANN으로 매칭
matches = flann.knnMatch(des_left, des_right, k=2)

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

for m, n in matches:
    #自适应阈值 , 자가 적응 임역
    # 좋은 matcing 걸러내기
    if n.distance > m.distance + disdif_avg:
        goodMatch.append(m)
        p2 = cv2.KeyPoint(kps_right[m.trainIdx][0],  kps_right[m.trainIdx][1],  1)
        p1 = cv2.KeyPoint(kps_left[m.queryIdx][0], kps_left[m.queryIdx][1], 1)
        locations_1_to_use.append([p1.pt[0], p1.pt[1]])
        locations_2_to_use.append([p2.pt[0], p2.pt[1]])
#goodMatch = sorted(goodMatch, key=lambda x: x.distance)
# 좋은 매칭으로 골라진 매칭점 개수 출력
print('match num is %d' % len(goodMatch))
locations_1_to_use = np.array(locations_1_to_use)
locations_2_to_use = np.array(locations_2_to_use)
# -------------------------------------------------------
# 좋은 매칭으로 골라진 매칭점으로 fundmental metrix 출력
F, mask = cv2.findFundamentalMat(locations_1_to_use,locations_2_to_use,cv2.FM_8POINT);
print('Fundamental Matrix is ')
print(F)
# F행렬 소수점 2번째 자리에서 반올림하기
print(np.round(F,2))
# --------------------------------------------------------

# Perform geometric verification using RANSAC.(기하학적 검증)
# 좋은 매칭으로 찾은 왼쪽, 오른쪽 특징점을 RANSAC을 이용하여 아핀변환으로 검증
_, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                          transform.AffineTransform,
                          min_samples=3,
                          residual_threshold=_RESIDUAL_THRESHOLD,
                          max_trials=1000)

print('Found %d inliers' % sum(inliers))
# inliner 변수 확인-----------------------
#print(inliers)
#----------------------------------------
'''
# --------------------------------------------------------
# ransac으로 골라진 매칭점으로 fundmental metrix 출력, 실패
F, mask = cv2.findFundamentalMat(locations_1_to_use,locations_2_to_use,cv2.FM_8POINT,inliers);
print('Fundamental Matrix is ')
print(F)
# F행렬 소수점 2번째 자리에서 반올림하기
# --------------------------------------------------------
'''
inlier_idxs = np.nonzero(inliers)[0]
#最终匹配结果 , 최종 일치 결과
matches = np.column_stack((inlier_idxs, inlier_idxs))
# 특징점 찾고 매칭하는 전체 시간
print('whole time is %6.3f' % (time.perf_counter() - start0))

# Visualize correspondences, and save to file.
#1 绘制匹配连线, 1 일치하는 연결 그리기
plt.rcParams['savefig.dpi'] = 100 #图片像素 , 이미지 픽셀
plt.rcParams['figure.dpi'] = 100 #分辨率 , 해상도
plt.rcParams['figure.figsize'] = (4.0, 3.0) # 设置figure_size尺寸 , figure_size 크기 설정
_, ax = plt.subplots()
plotmatch.plot_matches(
    ax,
    image1,
    image2,
    locations_1_to_use,
    locations_2_to_use,
    np.column_stack((inlier_idxs, inlier_idxs)),
    # 매칭 안된 포인트 안 그리기
    #plot_matche_points = False,
    # 매칭 안된 포인트 그리기
    plot_matche_points = True,
    matchline = True,
    matchlinewidth = 0.3)
ax.axis('off')
ax.set_title('')
plt.show()

