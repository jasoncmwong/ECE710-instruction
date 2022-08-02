from util.util import *
import numpy as np
import pandas as pd
import itertools as it
from sklearn.svm import SVC
import time


C_START = [0.1, 1.0, 10.0, 100.0]
GAMMA_START = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]


# EMPATICA
train_emp_x, train_emp_y, val_emp_x, val_emp_y, test_emp_x, test_emp_y = load_sets('empatica')

# Calculate weights for each class
w0, w1, w2 = calc_class_weights(train_emp_y)

# Normalize data
norm_train_emp_x, emp_mean, emp_std = normalize(train_emp_x)
norm_val_emp_x = (val_emp_x - emp_mean) / emp_std
norm_test_emp_x = (test_emp_x - emp_mean) / emp_std

# Run broad grid search to determine C and gamma to center search
# start = time.time()
# hp_grid = list(it.product(C_START, GAMMA_START))
# hp_results = pd.DataFrame(columns=['c', 'gamma', 'acc', 'out_percent', 'home_percent', 'sleep_percent'])
# for c, gamma in hp_grid:
#     svm = SVC(C=c,
#               kernel='rbf',
#               gamma=gamma,
#               class_weight={0: w0, 1: w1, 2: w2},
#               random_state=0)
#     svm.fit(norm_train_emp_x, train_emp_y)
#     val_pred = svm.predict(norm_val_emp_x)
#     acc, out_percent, home_percent, sleep_percent = calc_metrics(val_pred, val_emp_y)
#     hp_results = hp_results.append(pd.Series([c, gamma, acc, out_percent, home_percent, sleep_percent],
#                                              index=hp_results.columns),
#                                    ignore_index=True)
# opt_ind = np.argmax(hp_results['acc'])
# opt_param = hp_results.iloc[opt_ind][['c', 'gamma']]
# opt_acc = hp_results.iloc[opt_ind]['acc']
# grid_search_time = time.time() - start
# hp_results.to_csv('results/svm/empatica_svm-broad-hp-results.csv', index=False)

# PARAMETER VALUES TO CENTER SEARCH
# c: 10.0
# gamma: 0.1
# c_g = np.arange(2, 101)
# gamma_g = np.arange(0.02, 1, step=0.01)
# kernel_g = ['rbf', 'poly']

# Run focused grid search to optimize support vector machine parameters
# start = time.time()
# hp_grid = list(it.product(c_g, gamma_g, kernel_g))
# hp_results = pd.DataFrame(columns=['c', 'gamma', 'kernel', 'acc', 'out_percent', 'home_percent', 'sleep_percent'])
# for c, gamma, kernel in hp_grid:
#     svm = SVC(C=c,
#               kernel=kernel,
#               gamma=gamma,
#               class_weight={0: w0, 1: w1, 2: w2},
#               random_state=0)
#     svm.fit(norm_train_emp_x, train_emp_y)
#     val_pred = svm.predict(norm_val_emp_x)
#     acc, out_percent, home_percent, sleep_percent = calc_metrics(val_pred, val_emp_y)
#     hp_results = hp_results.append(pd.Series([c, gamma, kernel, acc, out_percent, home_percent, sleep_percent],
#                                              index=hp_results.columns),
#                                    ignore_index=True)
# opt_ind = np.argmax(hp_results['acc'])
# opt_param = hp_results.iloc[opt_ind][['c', 'gamma', 'kernel']]
# opt_acc = hp_results.iloc[opt_ind]['acc']
# grid_search_time = time.time() - start
# hp_results.to_csv('results/svm/empatica_svm-focused-hp-results.csv', index=False)

# RESULTS:
# OPTIMAL PARAMETERS
# c: 35.0
# gamma: 0.05
# kernel: rbf
# OPTIMAL VALIDATION PERFORMANCE
# acc: 81.6%
# out_percent: 80.0%
# home_percent: 76.2%
# sleep_percent: 88.9%
# SEARCH TIME: 6.2m

# Fit model on the training and validation set combined to get performance on the test set
final_train_emp_x = pd.concat((norm_train_emp_x, norm_val_emp_x)).reset_index(drop=True)
final_train_emp_y = pd.concat((train_emp_y, val_emp_y)).reset_index(drop=True)
w0, w1, w2 = calc_class_weights(final_train_emp_y)
emp_svm = SVC(C=35.0,
              kernel='rbf',
              gamma=0.05,
              class_weight={0: w0, 1: w1, 2: w2},
              random_state=0)
emp_svm.fit(final_train_emp_x, final_train_emp_y)
test_pred = emp_svm.predict(norm_test_emp_x)
emp_acc, emp_out_percent, emp_home_percent, emp_sleep_percent = calc_metrics(test_pred, test_emp_y)
conf_matrix = pd.crosstab(test_pred, test_emp_y)
conf_matrix = conf_matrix.rename(index={0: 'P Out', 1: 'P Home', 2: 'P Sleep'},
                                 columns={0: 'A Out', 1: 'A Home', 2: 'A Sleep'})
conf_matrix.to_csv('results/svm/empatica_svm-conf-matrix.csv')

# RESULTS:
# acc: 86.5%
# out_percent: 81.8%
# home_percent: 81.8%
# sleep_percent: 94.7%

# HEXOSKIN
train_hexo_x, train_hexo_y, val_hexo_x, val_hexo_y, test_hexo_x, test_hexo_y = load_sets('hexoskin')

# Calculate weights for each class
w0, w1, w2 = calc_class_weights(train_hexo_y)

# Normalize data
norm_train_hexo_x, hexo_mean, hexo_std = normalize(train_hexo_x)
norm_val_hexo_x = (val_hexo_x - hexo_mean) / hexo_std
norm_test_hexo_x = (test_hexo_x - hexo_mean) / hexo_std

# Run broad grid search to determine C and gamma to center search
# start = time.time()
# hp_grid = list(it.product(C_START, GAMMA_START))
# hp_results = pd.DataFrame(columns=['c', 'gamma', 'acc', 'out_percent', 'home_percent', 'sleep_percent'])
# for c, gamma in hp_grid:
#     svm = SVC(C=c,
#               kernel='rbf',
#               gamma=gamma,
#               class_weight={0: w0, 1: w1, 2: w2},
#               random_state=0)
#     svm.fit(norm_train_hexo_x, train_hexo_y)
#     val_pred = svm.predict(norm_val_hexo_x)
#     acc, out_percent, home_percent, sleep_percent = calc_metrics(val_pred, val_hexo_y)
#     hp_results = hp_results.append(pd.Series([c, gamma, acc, out_percent, home_percent, sleep_percent],
#                                              index=hp_results.columns),
#                                    ignore_index=True)
# opt_ind = np.argmax(hp_results['acc'])
# opt_param = hp_results.iloc[opt_ind][['c', 'gamma']]
# opt_acc = hp_results.iloc[opt_ind]['acc']
# grid_search_time = time.time() - start
# hp_results.to_csv('results/svm/hexoskin_svm-broad-hp-results.csv', index=False)

# PARAMETER VALUES TO CENTER SEARCH
# c: 10.0
# gamma: 0.1
# c_g = np.arange(2, 101)
# gamma_g = np.arange(0.02, 1, step=0.01)
# kernel_g = ['rbf', 'poly']

# Run focused grid search to optimize support vector machine parameters
# start = time.time()
# hp_grid = list(it.product(c_g, gamma_g, kernel_g))
# hp_results = pd.DataFrame(columns=['c', 'gamma', 'kernel', 'acc', 'out_percent', 'home_percent', 'sleep_percent'])
# for c, gamma, kernel in hp_grid:
#     svm = SVC(C=c,
#               kernel=kernel,
#               gamma=gamma,
#               class_weight={0: w0, 1: w1, 2: w2},
#               random_state=0)
#     svm.fit(norm_train_hexo_x, train_hexo_y)
#     val_pred = svm.predict(norm_val_hexo_x)
#     acc, out_percent, home_percent, sleep_percent = calc_metrics(val_pred, val_hexo_y)
#     hp_results = hp_results.append(pd.Series([c, gamma, kernel, acc, out_percent, home_percent, sleep_percent],
#                                              index=hp_results.columns),
#                                    ignore_index=True)
# opt_ind = np.argmax(hp_results['acc'])
# opt_param = hp_results.iloc[opt_ind][['c', 'gamma', 'kernel']]
# opt_acc = hp_results.iloc[opt_ind]['acc']
# grid_search_time = time.time() - start
# hp_results.to_csv('results/svm/hexoskin_svm-focused-hp-results.csv', index=False)

# RESULTS:
# OPTIMAL PARAMETERS
# c: 2.0
# gamma: 0.05
# kernel: rbf
# OPTIMAL VALIDATION PERFORMANCE
# acc: 89.4%
# out_percent: 71.4%
# home_percent: 85.7%
# sleep_percent: 100%
# SEARCH TIME: 5.9m

# Fit model on the training and validation set combined to get performance on the test set
final_train_hexo_x = pd.concat((norm_train_hexo_x, norm_val_hexo_x)).reset_index(drop=True)
final_train_hexo_y = pd.concat((train_hexo_y, val_hexo_y)).reset_index(drop=True)
w0, w1, w2 = calc_class_weights(final_train_hexo_y)
emp_svm = SVC(C=2.0,
              kernel='rbf',
              gamma=0.05,
              class_weight={0: w0, 1: w1, 2: w2},
              random_state=0)
emp_svm.fit(final_train_hexo_x, final_train_hexo_y)
test_pred = emp_svm.predict(norm_test_hexo_x)
hexo_acc, hexo_out_percent, hexo_home_percent, hexo_sleep_percent = calc_metrics(test_pred, test_hexo_y)
conf_matrix = pd.crosstab(test_pred, test_hexo_y)
conf_matrix = conf_matrix.rename(index={0: 'P Out', 1: 'P Home', 2: 'P Sleep'},
                                 columns={0: 'A Out', 1: 'A Home', 2: 'A Sleep'})
conf_matrix.to_csv('results/svm/hexoskin_svm-conf-matrix.csv')

# RESULTS:
# acc: 93.9%
# out_percent: 87.5%
# home_percent: 90.5%
# sleep_percent: 100%
print('Done')
