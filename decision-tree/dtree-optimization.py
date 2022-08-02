import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import itertools as it

MAX_DEPTH = [2, 3, 4]
MIN_NODE = [2, 4, 8, 16]
MIN_LEAF = [1, 2, 4, 8, 16]
MAX_LEAF = [2, 4, 8, 10, 12]

# Load data set
heart = pd.read_csv('heart.csv')

# Set input and output variables
x = heart[['num_vess', 'max_hr', 'chest_pain']]
y = heart['outcome']

# Get training and test sets
data_ind = pd.DataFrame(np.arange(len(heart)))
test_ind = np.ravel(data_ind.sample(n=int(np.round(len(heart) * 0.2)), random_state=0))
train_ind = [c for c in np.arange(len(heart)) if c not in test_ind]
train_x, train_y = x.iloc[train_ind], y.iloc[train_ind]
test_x, test_y = x.iloc[test_ind], y.iloc[test_ind]

hp_grid = list(it.product(MAX_DEPTH, MIN_NODE, MIN_LEAF, MAX_LEAF))
hp_results = pd.DataFrame(columns=['max_depth', 'min_node', 'min_leaf', 'max_leaf', 'acc', 'sens', 'spec'])
for max_depth, min_node, min_leaf, max_leaf in hp_grid:
    dtree = DecisionTreeClassifier(max_depth=max_depth,
                                   min_samples_split=min_node,
                                   min_samples_leaf=min_leaf,
                                   max_leaf_nodes=max_leaf,
                                   random_state=0).fit(train_x, train_y)
    pred = dtree.predict(test_x)
    tp = np.sum(np.logical_and(test_y == 1, pred == 1))
    tn = np.sum(np.logical_and(test_y == 0, pred == 0))
    fp = np.sum(np.logical_and(test_y == 0, pred == 1))
    fn = np.sum(np.logical_and(test_y == 1, pred == 0))
    acc = (tp + tn) / len(test_y)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    hp_results = hp_results.append(pd.Series([max_depth, min_node, min_leaf, max_leaf, acc, sens, spec],
                                             index=hp_results.columns),
                                   ignore_index=True)
opt_ind = np.argmax(hp_results['acc'])
print(hp_results.iloc[opt_ind])
print('Done')
