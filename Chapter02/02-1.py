# 02-1 훈련 세트와 테스트 세트


# 지도 학습과 비지도 학습
# 지도 학습에서 데이터와 정답은 입력과 타킷이라 하고, 합쳐 훈련 데이터라 함 -> 특성으로
# 알고리즘이 정답을 맞히는 것 : 지도학습, 맞힐 수 없음 : 비지도 학습

# 평가에 사용하는 데이터 : 테스트 세트, 훈련에 사용되는 데이터 : 훈련 세트


# 도미와 빙어의 데이터 리스트 , 하나의 생선 데이터 : 샘플
fish_length = [
  25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
  31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
  35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
  10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0
  ]

fish_weight = [
  242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
  500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
  700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
  7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9
  ]

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()

# fish_dat의 다섯번쨰 샘플 출력
print(fish_data[4])

# 슬라이싱, 마지막 인덱스는 포함 안됨 0~4까지 5개
print(fish_data[0:5])
print(fish_data[:5])
print(fish_data[44:])

# 훈련 세트로 입력값 중 0~34 인덱스까지 사용
train_input = fish_data[:35]
# 훈련 세트로 타깃 값 중 0~34 인덱스까지 사용
train_target = fish_target[:35]
# 테스트 세트로 입력 값 중 35~끝 인덱스까지 사용
test_input = fish_data[35:]
# 테스트 세트로 타킷 값 중 35~끝 인덱스까지 사용
test_target = fish_data[35:]

# 35개 샘플로 훈련 세트, 14개 샘플로 테스트 세트
kn = kn.fit(train_input, train_target) # 모델 훈련
# kn.score(test_input, test_target) # 평가

# 샘플링 편향 : 샘플링이 한 쪽으로 치우침 -> 간단하게 처리 : 넘파이

# 넘파이 : 배열 라이브러리, 고차원 배열 쉽게 사용
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

print(input_arr)

print(input_arr.shape) # (샘플 수, 특성 수)를 출력

np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)

print(index)

# 넘파이는 슬라이싱 이외에 배열 인덱싱 제공 : 1개의 인덱스가 아닌 여러개의 인덱스로 한 번에 여러개 원소 선택 가능
print(input_arr[[1,3]]) # input_arr에서 2,4번째 샘플 출력 가능

# index 배열의 처음 35개를 input_arr와 target_arr에 전달하여 랜덤하게 35개 샘플 훈련 세트 만들기
train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

print(input_arr[13], train_input[0]) # 두 값이 동일 index 첫번째 값이 13이기 때문

# 나머지 14개 세트
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# 훈련과 테스트 모두 잘 섞임을 확인 가능
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


# 두 번째 머신러닝 프로그램
# 훈련세트와 테스트세트로 k-최근접 이웃 모델을 훈련 시키자

kn = kn.fit(train_input, train_target)

kn.score(test_input, test_target) # 1.0, 100퍼센트 맞춤