import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

wine = pd.read_csv('https://bit.ly/wine_csv_data')

# Standardization
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# Preparing for training and test
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size = 0.2, random_state=42)

# Making validation set
sub_input, val_input, sub_target, val_taregt = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
print(sub_input.shape, sub_target.shape) # training: sub_, test: test_, validating: val_

# Training
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_taregt))

# k-fold cross validation (k=cv, default = 5)
scores = cross_validate(dt, train_input, train_target, cv = 5)
print(scores)

# Setting spliter to mix training set
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv = splitter)
print(np.mean(scores['test_score']))


# Hyperparameters tuning

params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)
print(gs.cv_results_['mean_test_score'])

best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])

params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))


# Random Search

# Scipy - uniform, randint instruction
rgen = randint(0, 10)
a = rgen.rvs(10)
print(a)
ugen = uniform(0, 1)
b = ugen.rvs(10)
print(b)

params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': range(2, 25),
          'min_samples_leaf': randint(1, 25)}

gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params,
                        n_iter = 100, n_jobs = -1, random_state = 42)
gs.fit(train_input, train_target)
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))

dt = gs.best_estimator_
print(dt.score(test_input, test_target))