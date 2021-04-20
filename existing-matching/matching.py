import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
import plotmatch
from skimage import measure
from skimage import transform


MIN_MATCH_COUNT = 10

imgfile1 = '../df-ms-data/1/df-googleearth-1k-20091227.jpg'
imgfile2 = '../df-ms-data/1/df-googleearth-1k-20181029.jpg'
imgfile1 = '../df-ms-data/1/df-uav-sar-500.jpg'

# 실험을 위해 임의로 추가
imgfile1 = '../df-ms-data/same_img/view1.png'
imgfile2 = '../df-ms-data/same_img/view5.png'

# 논문 비교 실험을 위해 추가
imgfile1 = '../df-ms-data/same_img/view1_1.png'
imgfile2 = '../df-ms-data/same_img/view5_1.png'

img1 = imageio.imread(imgfile1) # queryImage
img2 = imageio.imread(imgfile2) # trainImage
#img1 = cv2.imread(imgfile1,0) # queryImage
#img2 = cv2.imread(imgfile2,0) # trainImage

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
        
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)


img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()
'''


# --------------------------------------------
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


min_dist = 1000
max_dist = 0
disdif_avg = 0
# 매칭 점의 평균 거리차
for m, n in matches:
    disdif_avg += n.distance - m.distance
disdif_avg = disdif_avg / len(matches)

print(len(matches))
print(kp1[0])
print(type(kp1[0]))
print(matches[5][0].imgIdx)
print(type(matches[0][0]))

# cnn 방식으로 수정
for i,(m,n) in enumerate(matches):
    if n.distance > m.distance + disdif_avg:
        goodMatch.append(m)
        locations_2_to_use.append(kp2[m.trainIdx].pt)
        locations_1_to_use.append(kp1[m.queryIdx].pt)


# --------------------------------------------------------
# 좋은 매칭으로 골라진 매칭점 개수 출력
print('match num is %d' % len(goodMatch))
# 좋은 매칭으로 골라진 매칭점으로 fundmental metrix 출력
locations_1_to_use=np.asarray(locations_1_to_use)
locations_2_to_use=np.asarray(locations_2_to_use)
F, mask = cv2.findFundamentalMat(locations_1_to_use,locations_2_to_use,cv2.FM_8POINT);
print('Fundamental Matrix is ')
print(F)
# F행렬 소수점 2번째 자리에서 반올림하기
print(np.round(F,2))
# --------------------------------------------------------

# ransca
_, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                          transform.AffineTransform,
                          min_samples=3,
                          residual_threshold=30,
                          max_trials=1000)
inlier_idxs = np.nonzero(inliers)[0]
print('Found %d inliers' % sum(inliers))

# Visualize correspondences, and save to file.
#1 绘制匹配连线, 1 일치하는 연결 그리기
plt.rcParams['savefig.dpi'] = 100 #图片像素 , 이미지 픽셀
plt.rcParams['figure.dpi'] = 100 #分辨率 , 해상도
plt.rcParams['figure.figsize'] = (4.0, 3.0) # 设置figure_size尺寸 , figure_size 크기 설정
_, ax = plt.subplots()
plotmatch.plot_matches(
    ax,
    img1,
    img2,
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