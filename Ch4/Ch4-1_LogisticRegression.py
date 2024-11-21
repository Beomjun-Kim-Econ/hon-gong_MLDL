import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

# 데이터 준비
fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(fish.head())
print(pd.unique(fish['Species']))

# recognizable 하게 만들기
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])
fish_target = fish['Species'].to_numpy()
print(fish_target)

# 훈련데이터 / 홀드아웃셋으로 분리 및 standardization
train_input, test_input, train_target, test_target \
    = train_test_split(fish_input, fish_target, random_state=42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# k-NN으로 분류해보기
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

# 뜯어보기
print(kn.classes_)
print(kn.predict(test_scaled[:5]))
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))


## Logistic regression ###

# drawing sigmoid fucntion (phi = 1/(1+exp(-z))
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel = ('z')
plt.ylabel = ('phi')
plt.show()

## Logistic regression - binary classification
# 불리언 인덱싱은 생략
bream_smelt_indexing = (train_target == 'Bream') | (train_target == 'Smelt')
# | 은 or 이라는 의미 - 불리언 인덱싱에서는 | 을 써야한다.
train_bream_smelt = train_scaled[bream_smelt_indexing]
target_bream_smelt = train_target[bream_smelt_indexing]

# 훈련!
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))
print(lr.classes_)
print(lr.coef_, lr.intercept_)
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
print(expit(decisions))     # 위의 decisions에 대해 유니버셜 펑션 적용

## Logistic regression - multiple classification
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

# 뜯어보기
print(lr.predict(test_scaled[:5]))
proba2 = lr.predict_proba(test_scaled[:5])
print(np.round(proba2, decimals=3))
print(lr.classes_)

# 로지스틱 회귀 다중 분류 - softmax가 숨어있어요! - 생략

