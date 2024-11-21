import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

### SGDClassifier

# 데이터 준비
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target \
    = train_test_split(fish_input, fish_target, random_state=42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 훈련
sc = SGDClassifier(loss = 'log_loss', random_state=42, max_iter=1000)
sc.fit(train_scaled, train_target)
## .fit은 모든 데이터셋을 사용한다. 고로 sgdclassifer 에서 max_iter 은 에포크가 된다.
# print(sc.score(train_scaled, train_target))
# print(sc.score(test_scaled, test_target))

# 에포크
train_score = []
test_score = []
classes = np.unique(train_target)

for i in range(300) :
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))


plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 다시!
sc = SGDClassifier(loss = 'log_loss', random_state=42, max_iter=200, tol=None)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))



