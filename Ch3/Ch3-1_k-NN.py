import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

"""
k-NN regression
회귀: 최인접 k개의 mean of output variables
"""


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

# Visualization
plt.scatter(perch_length, perch_weight)
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# preparing training set and test set
train_input, test_input, train_target, test_target = \
    train_test_split(perch_length, perch_weight, random_state=42)

# training and test sets are sequence, not matrix(2-dimensional array), so we have to trasform it.
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
# print(train_input, test_input)


# 결정계수(Correlation Coefficient)와 훈련
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))

# SSE의 친구인 MAE(Mean among absolute values of errors)
test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
# 즉 mae는 test_target 과 test_prediction 사이의 차(=오차)를 절댓값하고 그것을 평균낸 것
print(mae)

# Overfitting and Underfitting
print(knr.score(train_input, train_target))
# score of train set(0.96988) < score of test set(0.99281) -> underfitting!

knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))