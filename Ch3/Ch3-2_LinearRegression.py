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

from sklearn.linear_model import LinearRegression

# 선형회귀로 훈련!
lr = LinearRegression()
lr. fit(train_input, train_target)
print(lr.predict([[50]]))          # 여러기를 동시에 물을수도 있으므로 array 로 반환
print(lr.coef_,lr.intercept_)      # 선형회귀의 기울기와 y절편 찾기 ; y_hat = m*x + b
# # 이 때 lr.coef_ 는 array 형태로 1*n 꼴이다. 왜냐, 다항회귀에서 여러개의 계수를 표현해야하므로!

# 선형회귀 시각화
plt.scatter(train_input, train_target)    # 훈련세트 산점도
plt.plot([15, 50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_])
# [15, 50]: x축 설정,[15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_] 은 두 점을 잇는 직선
plt.scatter(50, 1241.8, marker='^')       # 50cm 농어 표시
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

"""
Problem: 보니까 두 점수 너무 낮다. 과소적합된거네. 
Solution: 항을 추가하자. 다항회귀를 하자!!
2차 항을 추가하자. 기존 항을 제곱해서 새로운 열을 만들어주면 된다.
해당 배열을 만들자. 
"""
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
"""
위 두 줄은 np.column_stack 을 이용하여 어레이를 만든다. 
이때, np.column_stack 은 여러개의 nx1 배열(column)을 받아 어레이를 리턴한다.
"""
print(train_poly.shape, test_poly.shape)

lr_2 = LinearRegression()
lr_2.fit(train_poly, train_target)        # train_poly의 shape가 n*'2' -> 자동으로 2차식으로 회귀분석해준다.
print(lr_2.predict([[50**2, 50]]))
print(lr_2.coef_, lr_2.intercept_)
# [  1.01433211 -21.55792498] 116.05021078278338 -> 2차항 계수, 1차항 계수, 상수항 값이 각각 1.01, -21, 116이다.
# 이 때 lr_2.coef_ 는 array 형태로 1*n 꼴이다.

### 다항회귀 시각화 ###
point = np.arange(15, 50)   # 구간별 직선을 그리기 위해 15에서 49까지의 정수배열 만들기
plt.scatter(train_input, train_target)    # 훈련세트 산점도
plt.plot(point, lr_2.coef_[0] * (point**2) + lr_2.coef_[1] * point + lr_2.intercept_)
# 2차 방정식 그래프, 유저가 정수점을 찍어주면, matplot 이 알아서 곡선으로 바꿔준다.
plt.scatter(50, 1574, marker='^')       # 50cm 농어 표시
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
print(lr_2.score(train_poly, train_target))
print(lr_2.score(test_poly, test_target))