import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load data set
cancer = pd.read_csv('breast-cancer.csv')

# Balance dataset
num_neg = np.sum(cancer['diagnosis'] == 'B')
num_pos = np.sum(cancer['diagnosis'] == 'M')
num_remove = num_neg - num_pos
neg_ind = cancer.index[cancer['diagnosis'] == 'B']
data_ind = np.ravel(pd.DataFrame(np.arange(len(neg_ind))).sample(n=num_remove, random_state=0))
cancer = cancer.drop(neg_ind[data_ind])

# Set input and output variables
x = cancer[[c for c in cancer.columns if c != 'diagnosis']]
y = cancer['diagnosis']

# Get training and test sets
data_ind = pd.DataFrame(np.arange(len(cancer)))
test_ind = np.ravel(data_ind.sample(n=int(np.round(len(cancer) * 0.2)), random_state=0))
train_ind = [c for c in np.arange(len(cancer)) if c not in test_ind]
train_x, train_y = x.iloc[train_ind], y.iloc[train_ind]
test_x, test_y = x.iloc[test_ind], y.iloc[test_ind]

# Fit random forest
rforest = RandomForestClassifier(n_estimators=50,
                                 min_samples_leaf=2,
                                 max_features='auto',
                                 random_state=0).fit(train_x, train_y)

# Get feature importances
gini_importance = rforest.feature_importances_
print(x.columns)
print(gini_importance)

# Get performance metrics on test set
pred = rforest.predict(test_x)
tp = np.sum(np.logical_and(test_y == 'M', pred == 'M'))
tn = np.sum(np.logical_and(test_y == 'B', pred == 'B'))
fp = np.sum(np.logical_and(test_y == 'B', pred == 'M'))
fn = np.sum(np.logical_and(test_y == 'M', pred == 'B'))
acc = (tp + tn) / len(test_y)
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp)

dtree = DecisionTreeClassifier(random_state=97).fit(train_x, train_y)
pred = dtree.predict(test_x)
tp = np.sum(np.logical_and(test_y == 'M', pred == 'M'))
tn = np.sum(np.logical_and(test_y == 'B', pred == 'B'))
fp = np.sum(np.logical_and(test_y == 'B', pred == 'M'))
fn = np.sum(np.logical_and(test_y == 'M', pred == 'B'))
acc = (tp + tn) / len(test_y)
sens = tp / (tp + fn)
spec = tn / (tn + fp)
print('Done')
