import numpy as np
import pandas as pd
import itertools as it
from sklearn.svm import SVC

C = [4.0, 8.0, 10.0, 12.0, 16.0]
KERNEL = ['rbf']
GAMMA = ['scale', 'auto']

# Load data set
cancer = pd.read_csv('breast-cancer.csv')

# Balance dataset
num_neg = np.sum(cancer['diagnosis'] == 0)
num_pos = np.sum(cancer['diagnosis'] == 1)
num_remove = num_neg - num_pos
neg_ind = cancer.index[cancer['diagnosis'] == 0]
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

# Normalize input features
mean = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x = (train_x-mean)/std
test_x = (test_x-mean)/std

hp_grid = list(it.product(C, KERNEL, GAMMA))
hp_results = pd.DataFrame(columns=['c', 'kernel', 'gamma', 'acc', 'sens', 'spec'])
for c, kernel, gamma in hp_grid:
    svm = SVC(C=c,
              kernel=kernel,
              gamma=gamma,
              random_state=0)
    svm.fit(train_x, train_y)
    pred = svm.predict(test_x)
    tp = np.sum(np.logical_and(test_y == 1, pred == 1))
    tn = np.sum(np.logical_and(test_y == 0, pred == 0))
    fp = np.sum(np.logical_and(test_y == 0, pred == 1))
    fn = np.sum(np.logical_and(test_y == 1, pred == 0))
    acc = (tp + tn) / len(test_y)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    hp_results = hp_results.append(pd.Series([c, kernel, gamma, acc, sens, spec],
                                             index=hp_results.columns),
                                   ignore_index=True)
    print('Done: ', c, kernel, gamma)
opt_ind = np.argmax(hp_results['acc'])
print(hp_results.iloc[opt_ind])
print('Done')
