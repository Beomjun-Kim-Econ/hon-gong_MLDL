import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

### repeating Ch3-1 ###
# Dataset
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# preparing training set and test set
train_input, test_input, train_target, test_target = \
    train_test_split(perch_length, perch_weight, random_state=42)

# training and test sets are sequence, not matrix(2-dimensional array), so we have to trasform it.
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)

### repeating have finished
print(knr.predict([[50]]))  # 1033 ; 실제값과 차이가 많이난다?!

# 50cm 농어의 이웃:
distance, indexes = knr.kneighbors([[50]])
# 여기서 indexes 는 50cm 농어의 이웃 3개의 인덱스 배열
# distance 는 그 이웃과 50cm 농어 사이의 거리 배열

# 훈련세트의 산점도
plt.scatter(train_input, train_target)

# 훈련세트 중 50cm 농어의 이웃 샘플만 다시 그리기
plt.scatter(train_input[indexes], train_target[indexes], marker='D')

plt.scatter(50, 1033, marker= '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


"""
Problem: 흠... 길이가 이웃 3개보다 더 큰데 무게가 이상하다!
Solution: Linear Regression
"""
