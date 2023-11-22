# 01-3 마켓과 머신러닝


# 생선 분류 문제

# 도미 데이터 준비하기 -> 특성
bream_length = [
  25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7,
  31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5,
  34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0,
  38.5, 38.5, 39.5, 41.0, 41.0
]

bream_weight = [
  242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0,
  450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0,
  700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
  700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0,
  925.0, 975.0, 950.0
]

# x, y 를 그래프로 표현한 것 : 산점도 -> matplotlib(맷플롯립)
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


# 빙어 데이터 준비하기
smelt_length = [
  9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2,
  12.4, 13.0, 14.3, 15.0
]

smelt_weight = [
  6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 
  13.4, 12.2, 19.7, 19.9
]

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


# 첫 번째 머신러닝 프로그램
# k-최근접 이웃 알고리즘을 사용해 도미, 빙어 데이터 구분
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# scikit-learn(사이킷런) 머신러닝 패기지 사용하려면 세로 방향으로 2차원 리스트 만들어야 함 -> zip() 이용
fish_data = [[l, w] for l, w in zip(length, weight)]

print(fish_data) # 2차원 리스트 = 리스트의 리스트

# 도미와 빙어를 1과 0으로 표현해서 구분
fish_target = [1] * 35 + [0] * 14
print(fish_target)

# 사이킷린 k-최근접 이웃 알고리즘 구현 클래스 임포트
# import sklearn
# model = sklearn.neighbors.KNeighborsClassifier()
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier() # 객체 생성
# fish_target 정보 전달해 도미 찾기 위한 기준을 학습 시킴 = 훈련
kn.fit(fish_data, fish_target)

# 정확도, 0~1 값 반환
kn.score(fish_data, fish_target)

# 새로운 데이터의 정답을 예측
kn.predict([[30, 600]]) # array([1]) 반환 = 도미 예측

# k-최근접 이웃 알고리즘 : 새로운 데이터 예측할 때는 가장 가까운 직선거리에 어떤 데이터 있는지
# 단점 : 데이터가 아주 많은 경우 사용 어렵, 메모리 많이 필요하고 직선거리 계산하는데도 많은 시간 소요

print(kn._fix_X) # 2차원 배열(fish_data)
print(kn._y) # fish_target 배열
# k-최근접 이웃 알고리즘은 훈련되는 것이 없고, 데이터 저장 후 새로운 데이터 등장하면 가까운 데이터 참고

# 참고할 가까운 데이터의 갯수 지정
kn49 = KNeighborsClassifier(n_neighbors=49) # 참고 데이터 49개로 한 모델
# 가장 가까운 데이터 49개 사용, but fish_data 갯수가 49개
# 49개 중 도미가 35개로 다수이므로 어떤 데이터를 넣어도 도미를 예측함

kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target) # 35/49 -> 49개 중 도미 35개이기 때문
