import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

"""
Exploratory data analysis
"""
wine = pd.read_csv('https://bit.ly/wine_csv_data')
print(wine.head())
print(wine.info())
print(wine.describe())

"""
회귀로 분류를 시도해보자...
"""
# Standardization
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# Preparing for training and test
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size = 0.2, random_state=42)
print(train_input.shape, train_target.shape)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# training
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

# Result function of logistic regression
print(lr.coef_, lr.intercept_)

"""
결정트리 도입
"""
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))       # Overfitted

# 결정트리 시각화
# plt.figure(figsize = (10, 7))
# plot_tree(dt)
# plt.show()

plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alchohol', 'sugar', 'pH'])
plt.show()

# Regulation - limit max depth
dt = DecisionTreeClassifier(random_state=42, max_depth=3)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

plt.figure(figsize=(10, 7))
plot_tree(dt, filled=True, feature_names = ['alcohol', 'sugar', 'pH'])
plt.show()

# 사실 결정트리는 굳이 전처리(standardization)할 필요가 없다. 왜? 불순도, ratio에 scale은 없다...
print(dt.feature_importances_)