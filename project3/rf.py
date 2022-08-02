import pandas as pd
import numpy as np
from util.util import *
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time
import itertools as it

NUM_TREES_START = [2, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200]


def num_tree_search(train_x, train_y, val_x, val_y, weights=None):
    accs = []
    for num_trees in NUM_TREES_START:
        rforest = RandomForestClassifier(n_estimators=num_trees,
                                         max_features='auto',
                                         class_weight=weights,
                                         random_state=0).fit(train_x, train_y)
        val_pred = rforest.predict(val_x)
        acc, out_percent, home_percent, sleep_percent = calc_metrics(val_pred, val_y)
        accs.append(acc)

    # Plot validation accuracy vs. number of trees to find point of accuracy stagnating
    plt.plot(NUM_TREES_START, accs)
    plt.show()


# EMPATICA
train_emp_x, train_emp_y, val_emp_x, val_emp_y, test_emp_x, test_emp_y = load_sets('empatica')

# Calculate weights for each class
w0, w1, w2 = calc_class_weights(train_emp_y)

# Fit random forests over multiple numbers of trees to start to estimate optimal number of trees
# num_tree_search(train_emp_x, train_emp_y, val_emp_x, val_emp_y, {0: w0, 1: w1, 2: w2})

# RESULTS: optimal performance at 50 trees

# PARAMETER VALUES TO CENTER SEARCH:
# n_estimators: 50
# m: 3
# num_trees_g = np.arange(25, 100, step=25)
# m_g = np.arange(2, 6)
# max_depth_g = [2, 3, 4, 5, None]
# num_leaf_g = [1, 2, 4, 8, 16]
# num_split_g = [2, 4, 8, 16, 32]

# Run grid search to optimize random forest parameters
# start = time.time()
# hp_grid = list(it.product(num_trees_g, m_g, max_depth_g, num_leaf_g, num_split_g))
# hp_results = pd.DataFrame(columns=['num_trees', 'm', 'max_depth', 'num_leaf', 'num_split', 'acc', 'out_percent',
#                                    'home_percent', 'sleep_percent'])
# for num_trees, m, max_depth, num_leaf, num_split in hp_grid:
#     rforest = RandomForestClassifier(n_estimators=num_trees,
#                                      max_features=m,
#                                      max_depth=max_depth,
#                                      min_samples_leaf=num_leaf,
#                                      min_samples_split=num_split,
#                                      class_weight={0: w0, 1: w1, 2: w2},
#                                      random_state=0).fit(train_emp_x, train_emp_y)
#     val_pred = rforest.predict(val_emp_x)
#     acc, out_percent, home_percent, sleep_percent = calc_metrics(val_pred, val_emp_y)
#     hp_results = hp_results.append(pd.Series([num_trees, m, max_depth, num_leaf, num_split, acc, out_percent,
#                                               home_percent, sleep_percent],
#                                              index=hp_results.columns),
#                                    ignore_index=True)
# opt_ind = np.argmax(hp_results['acc'])
# opt_param = hp_results.iloc[opt_ind][['num_trees', 'm', 'max_depth', 'num_leaf', 'num_split']]
# opt_acc = hp_results.iloc[opt_ind]['acc']
# grid_search_time = time.time() - start
# hp_results.to_csv('results/rf/empatica_rf-hp-results.csv', index=False)

# RESULTS:
# OPTIMAL PARAMETERS
# num_trees: 25
# m: 4
# max_depth: 3
# num_leaf: 1
# num_split: 32
# OPTIMAL VALIDATION PERFORMANCE
# acc: 87.8%
# out_percent: 81.8%
# home_percent: 88.9%
# sleep_percent: 94.1%
# SEARCH TIME: 2.8m

# Fit model on the training and validation set combined to get performance on the test set
final_train_emp_x = pd.concat((train_emp_x, val_emp_x)).reset_index(drop=True)
final_train_emp_y = pd.concat((train_emp_y, val_emp_y)).reset_index(drop=True)
w0, w1, w2 = calc_class_weights(final_train_emp_y)
emp_rf = RandomForestClassifier(n_estimators=25,
                                max_features=4,
                                max_depth=3,
                                min_samples_leaf=1,
                                min_samples_split=32,
                                class_weight={0: w0, 1: w1, 2: w2},
                                random_state=0).fit(final_train_emp_x, final_train_emp_y)
test_pred = emp_rf.predict(test_emp_x)
emp_acc, emp_out_percent, emp_home_percent, emp_sleep_percent = calc_metrics(test_pred, test_emp_y)
conf_matrix = pd.crosstab(test_pred, test_emp_y)
conf_matrix = conf_matrix.rename(index={0: 'P Out', 1: 'P Home', 2: 'P Sleep'},
                                 columns={0: 'A Out', 1: 'A Home', 2: 'A Sleep'})
conf_matrix.to_csv('results/rf/empatica_rf-conf-matrix.csv')
feat_imp = sorted(zip(train_emp_x.columns, emp_rf.feature_importances_), key=lambda l: l[1], reverse=True)
imp_str = ''
for i in range(len(feat_imp)):
    imp_str += '{0}: {1:.3f}\n'.format(feat_imp[i][0], feat_imp[i][1])
print(imp_str)

# RESULTS:
# acc: 92.3%
# out_percent: 90.9%
# home_percent: 90.9%
# sleep_percent: 94.7%

# HEXOSKIN
train_hexo_x, train_hexo_y, val_hexo_x, val_hexo_y, test_hexo_x, test_hexo_y = load_sets('hexoskin')

# Calculate weights for each class
w0, w1, w2 = calc_class_weights(train_hexo_y)

# Fit random forests over multiple numbers of trees to start to estimate optimal number of trees
# num_tree_search(train_hexo_x, train_hexo_y, val_hexo_x, val_hexo_y, {0: w0, 1: w1, 2: w2})

# RESULTS: stagnant throughout all numbers of trees - use optimal number from Empatica since data sets are similar

# PARAMETER VALUES TO CENTER SEARCH:
# n_estimators: 50
# m: 3
# num_trees_g = np.arange(25, 200, step=25)
# m_g = np.arange(2, 6)
# max_depth_g = [2, 3, 4, 5, None]
# num_leaf_g = [1, 2, 4, 8, 16]
# num_split_g = [2, 4, 8, 16, 32]

# Run grid search to optimize random forest parameters
# start = time.time()
# hp_grid = list(it.product(num_trees_g, m_g, max_depth_g, num_leaf_g, num_split_g))
# hp_results = pd.DataFrame(columns=['num_trees', 'm', 'max_depth', 'num_leaf', 'num_split', 'acc', 'out_percent',
#                                    'home_percent', 'sleep_percent'])
# for num_trees, m, max_depth, num_leaf, num_split in hp_grid:
#     rforest = RandomForestClassifier(n_estimators=num_trees,
#                                      max_features=m,
#                                      max_depth=max_depth,
#                                      min_samples_leaf=num_leaf,
#                                      min_samples_split=num_split,
#                                      class_weight={0: w0, 1: w1, 2: w2},
#                                      random_state=0).fit(train_emp_x, train_emp_y)
#     val_pred = rforest.predict(val_emp_x)
#     acc, out_percent, home_percent, sleep_percent = calc_metrics(val_pred, val_emp_y)
#     hp_results = hp_results.append(pd.Series([num_trees, m, max_depth, num_leaf, num_split, acc, out_percent,
#                                               home_percent, sleep_percent],
#                                              index=hp_results.columns),
#                                    ignore_index=True)
# opt_ind = np.argmax(hp_results['acc'])
# opt_param = hp_results.iloc[opt_ind][['num_trees', 'm', 'max_depth', 'num_leaf', 'num_split']]
# opt_acc = hp_results.iloc[opt_ind]['acc']
# grid_search_time = time.time() - start
# print(opt_acc, grid_search_time)
# hp_results.to_csv('results/rf/hexoskin_rf-hp-results.csv', index=False)

# RESULTS:
# OPTIMAL PARAMETERS
# num_trees: 50
# m: 5
# max_depth: None
# num_leaf: 8
# num_split: 2
# OPTIMAL VALIDATION PERFORMANCE
# acc: 85.7%
# out_percent: 75.0%
# home_percent: 85.0%
# sleep_percent: 94.1%
# SEARCH TIME: 12.1m

# Fit model on the training and validation set combined to get performance on the test set
final_train_hexo_x = pd.concat((train_hexo_x, val_hexo_x)).reset_index(drop=True)
final_train_hexo_y = pd.concat((train_hexo_y, val_hexo_y)).reset_index(drop=True)
w0, w1, w2 = calc_class_weights(final_train_hexo_y)
hexo_rf = RandomForestClassifier(n_estimators=50,
                                 max_features=5,
                                 max_depth=None,
                                 min_samples_leaf=8,
                                 min_samples_split=2,
                                 class_weight={0: w0, 1: w1, 2: w2},
                                 random_state=0).fit(final_train_hexo_x, final_train_hexo_y)
test_pred = hexo_rf.predict(test_hexo_x)
hexo_acc, hexo_out_percent, hexo_home_percent, hexo_sleep_percent = calc_metrics(test_pred, test_hexo_y)
conf_matrix = pd.crosstab(test_pred, test_hexo_y)
conf_matrix = conf_matrix.rename(index={0: 'P Out', 1: 'P Home', 2: 'P Sleep'},
                                 columns={0: 'A Out', 1: 'A Home', 2: 'A Sleep'})
conf_matrix.to_csv('results/rf/hexoskin_rf-conf-matrix.csv')
feat_imp = sorted(zip(train_hexo_x.columns, hexo_rf.feature_importances_), key=lambda l: l[1], reverse=True)
imp_str = ''
for i in range(len(feat_imp)):
    imp_str += '{0}: {1:.3f}\n'.format(feat_imp[i][0], feat_imp[i][1])
print(imp_str)

# RESULTS:
# acc: 89.8%
# out_percent: 87.5%
# home_percent: 85.7%
# sleep_percent: 95.0%
print('Done')
