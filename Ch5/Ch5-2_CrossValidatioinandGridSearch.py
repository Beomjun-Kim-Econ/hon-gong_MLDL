import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

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