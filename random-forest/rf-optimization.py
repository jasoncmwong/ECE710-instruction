import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import itertools as it

NUM_TREES = [10, 25, 50, 100, 200]
MIN_LEAF = [2, 4, 8, 16, 32]
MAX_FEAT = ['auto']

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

hp_grid = list(it.product(NUM_TREES, MIN_LEAF, MAX_FEAT))
hp_results = pd.DataFrame(columns=['num_trees', 'min_leaf', 'max_feat', 'acc', 'sens', 'spec'])
for num_trees, min_leaf, max_feat in hp_grid:
    rforest = RandomForestClassifier(n_estimators=num_trees,
                                     min_samples_leaf=min_leaf,
                                     max_features=max_feat,
                                     random_state=0).fit(train_x, train_y)
    pred = rforest.predict(test_x)
    tp = np.sum(np.logical_and(test_y == 'M', pred == 'M'))
    tn = np.sum(np.logical_and(test_y == 'B', pred == 'B'))
    fp = np.sum(np.logical_and(test_y == 'B', pred == 'M'))
    fn = np.sum(np.logical_and(test_y == 'M', pred == 'B'))
    acc = (tp + tn) / len(test_y)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    hp_results = hp_results.append(pd.Series([num_trees, min_leaf, max_feat, acc, sens, spec],
                                             index=hp_results.columns),
                                   ignore_index=True)
    print('Done: ', num_trees, min_leaf, max_feat)
opt_ind = np.argmax(hp_results['acc'])
print(hp_results.iloc[opt_ind])
print('Done')
