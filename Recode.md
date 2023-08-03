## 이미지 input
train.py 
eval.py 
Average PCE: 0.6778571428571428
-> Average PCE: 0.695


## 2D pose - 1
normalize_screen_coordinates
2D pose는 screen 좌표계 정규화를 한다.
2D pose만 입력으로 받는다.
train_with_pose.py 
eval_with_pose.py
Average PCE: 0.7125
-> Average PCE: 0.7296428571428571
-> Average PCE: 0.7321428571428571

## 2D pose - 2
normalize_screen_coordinates
2D pose는 screen 좌표계 정규화를 한다.
2D pose+conf를 입력으로 받는다.
train_with_pose1.py 
eval_with_pose1.py
Average PCE: 0.7228571428571429
-> Average PCE: 0.7246428571428571
-> Average PCE: 0.7332142857142857

## 2D pose - 3
normalize_CanonPose
CanonPose에서 소개된 2D pose 정규화를 수행한다 
2D pose만 입력으로 받는다.
train_with_pose5.py 
eval_with_pose5.py
Average PCE: 0.7082142857142857
-> Average PCE: 0.7232142857142857
-> Average PCE: 0.7210714285714286

## 2D pose - 4
normalize_CanonPose
CanonPose에서 소개된 2D pose 정규화를 수행한다 
2D pose+conf를 입력으로 받는다.
train_with_pose6.py 
eval_with_pose6.py
Average PCE: 0.7264285714285714
-> Average PCE: 0.72
-> Average PCE: 0.7264285714285714

## 2D pose - 5
mode 1 + resblock
Average PCE: 0.7642857142857142

## 2D pose - 6
mode 2 + resblock
Average PCE: 0.7560714285714286

## 2D pose - 7
mode 3 + resblock
Average PCE: 0.7635714285714286

## 2D pose - 7
mode 4 + resblock
Average PCE: 0.7546428571428572


## 3D pose 
PoseFormerV2 

Average PCE: 0.7403571428571428



