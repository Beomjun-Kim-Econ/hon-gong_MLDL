import numpy as np
import pandas as pd


df = pd.read_csv('https://bit.ly/perch_csv_data')   # 데이터 프레임(py_da 5장 참고)
perch_full = df.to_numpy()      # 넘파이 어레이로 변환

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])     # 농어 무게 어레이

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target \
    = train_test_split(perch_full, perch_weight, random_state = 42)

# PolynomialFeatures 를 살펴보자
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)        # default degree = 2
poly.fit([[2,3]])    # 특성 모두 만들기 | 학습만 하므로 반환하는 건 없다. None을 반환.
"""
PolynomialFeatures.fit에 대한 설명
poly.fit 의 역할은 (1) 차수 결정(degree=2) (2) include_bias 결정(False, True)
즉, 위 코드에서 [[2,3]]은 그저 데이터 구조를 알려주는 것. 여기서 데이터 구조란 열의 개수(=행의 길이 = 피쳐의 개수)
아래에서 poly.fit 에 훈련데이터셋을 넣었는데 굳이 그럴 필요는 없다는 것.
내가 훈련데이터셋의 열의 개수를 안다면 그 길이에 맞는 아무 [[1,2,3,...,n]]을 넣어줘도 된다!
!!! 하지만 아무거나 넣으면 당연히 혼란이 온다. 고로 제발 좀 훈련데이터셋으로 넣자 !!!
"""
print(poly.transform([[3,4]]))

# 이제 적용해보자.
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(poly.get_feature_names_out())
test_poly = poly.transform(test_input)

# 다중 회귀 모델로 훈련해보자
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))

# 규제 - 릿지와 라쏘
# 피쳐의 스케일이 정규화(standardization)가 되어야 한다.
from sklearn.preprocessing import StandardScaler        # 정규화 해주는 클래스

ss = StandardScaler()
ss.fit(train_poly)          # ss 안에 .fit(train_poly)가 새로운 객체로 저장됨
train_scaled = ss.transform(train_poly)          # 이걸 꺼냄
test_scaled = ss.transform(test_poly)

# 릿지 회귀
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)    # alpha 설정은 여기서. 기본값은 1.0
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# finding the most apt alpha
# 알파를 여러개 넣어보며 R^2 값의 변화를 보자. - 시각화해봅시다.
import matplotlib.pyplot as plt

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list :
       ridge = Ridge(alpha=alpha)
       ridge.fit(train_scaled, train_target)
       train_score.append(ridge.score(train_scaled, train_target))
       test_score.append(ridge.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)      # np.log10(alpha_list) 가 x축, train_score가 y축
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# 위의 그래프에서 alpha = 0.1 일 때가 가장 좋음을 알았다.
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# 라쏘회귀도 보자
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

# 마찬가지로 alpha 리스트를 보자
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list :
       lasso = Lasso(alpha=alpha, max_iter=1000000)
       lasso.fit(train_scaled, train_target)
       train_score.append(lasso.score(train_scaled, train_target))
       test_score.append(lasso.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)      # np.log10(alpha_list) 가 x축, train_score가 y축
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# 최적의 알파는 log10(n)= 1, 즉 10 임을 알 수 있다.
lasso = Lasso(alpha=1, max_iter=1000000)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
print(np.sum(lasso.coef_ == 0))