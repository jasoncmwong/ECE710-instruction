from util.util import *
import numpy as np
import pandas as pd
import itertools as it
import tensorflow as tf
import time

NUM_CLASSES = 3
FIRST_NNEURONS_START = [3, 5, 7]
SECOND_NNEURONS_START = [3, 5, 7]
LR_START = [1e-4, 1e-3, 1e-2, 1e-1]
DROPOUT_START = [0, 0.5]
BATCH_NORM_START = [True, False]
BATCH_SIZE_START = [5, 47]
NUM_EPOCHS_START = [25, 50, 100, 200, 400]


def one_hot_encode(targets):
    lbls = train_emp_y.unique()
    ohe_targets = pd.DataFrame(data=np.zeros((len(targets), len(lbls)), dtype=bool))
    for i in lbls:
        ohe_targets[i].loc[targets == i] = True
    return ohe_targets


def create_ann(first_nneurons, second_nneurons, act_fcn, lr, dropout, batch_norm):
    ann = tf.keras.Sequential()
    ann.add(tf.keras.layers.Dense(first_nneurons, activation=act_fcn))
    if batch_norm:
        ann.add(tf.keras.layers.BatchNormalization())
    ann.add(tf.keras.layers.Dropout(dropout))
    if second_nneurons > 0:
        ann.add(tf.keras.layers.Dense(second_nneurons))
        if batch_norm:
            ann.add(tf.keras.layers.BatchNormalization())
        ann.add(tf.keras.layers.Dropout(dropout))
    ann.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    ann.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                weighted_metrics=['accuracy'])
    return ann


# EMPATICA
train_emp_x, train_emp_y, val_emp_x, val_emp_y, test_emp_x, test_emp_y = load_sets('empatica')

# Calculate weights for each class
w0, w1, w2 = calc_class_weights(train_emp_y)

# Normalize data
norm_train_emp_x, emp_mean, emp_std = normalize(train_emp_x)
norm_val_emp_x = (val_emp_x - emp_mean) / emp_std
norm_test_emp_x = (test_emp_x - emp_mean) / emp_std

# One-hot encode targets
ohe_train_emp_y = one_hot_encode(train_emp_y)
ohe_val_emp_y = one_hot_encode(val_emp_y)
ohe_test_emp_y = one_hot_encode(test_emp_y)

# Run broad grid search to determine hyperparameters to center search
# start = time.time()
# hp_grid = list(it.product(FIRST_NNEURONS_START, SECOND_NNEURONS_START, LR_START, DROPOUT_START, BATCH_NORM_START,
#                           BATCH_SIZE_START, NUM_EPOCHS_START))
# hp_results = pd.DataFrame(columns=['first_nneurons', 'second_nneurons', 'act_fcn', 'lr', 'dropout', 'batch_norm',
#                                    'batch_size', 'num_epochs', 'acc', 'out_percent', 'sleep_percent', 'home_percent'])
# for first_nneurons, second_nneurons, lr, dropout, batch_norm, batch_size, num_epochs in hp_grid:
#     emp_ann = create_ann(first_nneurons, second_nneurons, 'sigmoid', lr, dropout, batch_norm)
#     ann_fit = emp_ann.fit(x=norm_train_emp_x,
#                           y=ohe_train_emp_y,
#                           batch_size=batch_size,
#                           epochs=num_epochs,
#                           validation_data=(norm_val_emp_x, ohe_val_emp_y),
#                           class_weight={0: w0, 1: w1, 2: w2})
#     val_pred = np.array([np.argmax(c) for c in emp_ann.predict(norm_val_emp_x)], dtype=float)
#     acc, out_percent, home_percent, sleep_percent = calc_metrics(val_pred, val_emp_y)
#     hp_results = hp_results.append(pd.Series([first_nneurons, second_nneurons, 'sigmoid', lr, dropout, batch_norm,
#                                              batch_size, num_epochs, acc, out_percent, home_percent, sleep_percent],
#                                              index=hp_results.columns),
#                                    ignore_index=True)
# opt_ind = np.argmax(hp_results['acc'])
# opt_param = hp_results.iloc[opt_ind][['first_nneurons', 'second_nneurons', 'act_fcn', 'lr', 'dropout', 'batch_norm',
#                                       'batch_size', 'num_epochs']]
# opt_acc = hp_results.iloc[opt_ind]['acc']
# grid_search_time = time.time() - start
# hp_results.to_csv('results/ann/empatica_ann-broad-hp-results.csv', index=False)

# PARAMETER VALUES TO CENTER SEARCH
# first_nneurons: 5
# second_nneurons: 5
# lr: 0.01
# dropout: 0
# batch_norm: True
# batch_size: 47
# num_epochs: 200
# OPTIMAL VALIDATION PERFORMANCE
# acc: 85.7%
# out_percent: 90.0%
# home_percent: 88.9%
# sleep_percent: 81.0%
# SEARCH TIME: 2.8h

# Run focused grid search to optimize ANN parameters
# first_nneurons_g = [4, 5, 6]
# second_nneurons_g = [4, 5, 6]
# act_fcn_g = ['tanh', 'sigmoid']
# lr_g = [1e-3, 1e-2, 1e-1]
# dropout_g = [0, 0.125]
# batch_norm_g = [True, False]
# batch_size_g = [5, 47]
# num_epochs_g = [150, 200, 250, 300, 350]

# Run grid search to optimize ANN parameters
# start = time.time()
# hp_grid = list(it.product(first_nneurons_g, second_nneurons_g, act_fcn_g, lr_g, dropout_g, batch_norm_g, batch_size_g,
#                           num_epochs_g))
# hp_results = pd.DataFrame(columns=['first_nneurons', 'second_nneurons', 'act_fcn', 'lr', 'dropout', 'batch_norm',
#                                    'batch_size', 'num_epochs', 'acc', 'out_percent', 'home_percent', 'sleep_percent'])
# for first_nneurons, second_nneurons, act_fcn, lr, dropout, batch_norm, batch_size, num_epochs in hp_grid:
#     emp_ann = create_ann(first_nneurons, second_nneurons, act_fcn, lr, dropout, batch_norm)
#     ann_fit = emp_ann.fit(x=norm_train_emp_x,
#                           y=ohe_train_emp_y,
#                           batch_size=batch_size,
#                           epochs=num_epochs,
#                           validation_data=(norm_val_emp_x, ohe_val_emp_y),
#                           class_weight={0: w0, 1: w1, 2: w2})
#     val_pred = np.array([np.argmax(c) for c in emp_ann.predict(norm_val_emp_x)], dtype=float)
#     acc, out_percent, home_percent, sleep_percent = calc_metrics(val_pred, val_emp_y)
#     hp_results = hp_results.append(pd.Series([first_nneurons, second_nneurons, act_fcn, lr, dropout, batch_norm,
#                                              batch_size, num_epochs, acc, out_percent, home_percent, sleep_percent],
#                                              index=hp_results.columns),
#                                    ignore_index=True)
# opt_ind = np.argmax(hp_results['acc'])
# opt_param = hp_results.iloc[opt_ind][['first_nneurons', 'second_nneurons', 'act_fcn', 'lr', 'dropout', 'batch_norm',
#                                       'batch_size', 'num_epochs']]
# opt_acc = hp_results.iloc[opt_ind]['acc']
# grid_search_time = time.time() - start
# hp_results.to_csv('results/ann/empatica_ann-focused-hp-results.csv', index=False)

# RESULTS:
# OPTIMAL PARAMETERS
# first_nneurons: 4
# second_nneurons: 6
# act_fcn: 'tanh'
# lr: 0.01
# dropout: 0
# batch_norm: True
# batch_size: 47
# num_epochs: 200
# OPTIMAL VALIDATION PERFORMANCE
# acc: 85.7%
# out_percent: 80.0%
# home_percent: 81.0%
# sleep_percent: 94.4%
# SEARCH TIME: 6.1h

# Fit model on the training set, using the validation set to get best model, and get performance on the test set
# chk_cbk = tf.keras.callbacks.ModelCheckpoint('results/ann/empatica_ann.h5',  # Model checkpoint callback
#                                              monitor='val_loss',
#                                              verbose=True,
#                                              save_best_only=True,
#                                              mode='min')
# emp_ann = create_ann(4, 6, 'tanh', 0.01, 0, True)
# ann_fit = emp_ann.fit(x=norm_train_emp_x,
#                       y=ohe_train_emp_y,
#                       batch_size=47,
#                       epochs=200,
#                       validation_data=(norm_val_emp_x, ohe_val_emp_y),
#                       class_weight={0: w0, 1: w1, 2: w2},
#                       callbacks=[chk_cbk])
# val_curve = pd.DataFrame(ann_fit.history['val_loss'])
# val_curve.to_csv('results/ann/empatica_val_curve.csv')
# emp_ann = tf.keras.models.load_model('results/ann/empatica_ann.h5')
# test_pred = np.array([np.argmax(c) for c in emp_ann.predict(norm_test_emp_x)], dtype=float)
# emp_acc, emp_out_percent, emp_home_percent, emp_sleep_percent = calc_metrics(test_pred, test_emp_y)
# conf_matrix = pd.crosstab(test_pred, test_emp_y)
# conf_matrix = conf_matrix.rename(index={0: 'P Out', 1: 'P Home', 2: 'P Sleep'},
#                                  columns={0: 'A Out', 1: 'A Home', 2: 'A Sleep'})
# conf_matrix.to_csv('results/ann/empatica_ann-conf-matrix.csv')

# RESULTS:
# acc: 88.5%
# out_percent: 63.6%
# home_percent: 95.5%
# sleep_percent: 94.7%

# HEXOSKIN
train_hexo_x, train_hexo_y, val_hexo_x, val_hexo_y, test_hexo_x, test_hexo_y = load_sets('hexoskin')

# Calculate weights for each class
w0, w1, w2 = calc_class_weights(train_hexo_y)

# Normalize data
norm_train_hexo_x, hexo_mean, hexo_std = normalize(train_hexo_x)
norm_val_hexo_x = (val_hexo_x - hexo_mean) / hexo_std
norm_test_hexo_x = (test_hexo_x - hexo_mean) / hexo_std

# One-hot encode targets
ohe_train_hexo_y = one_hot_encode(train_hexo_y)
ohe_val_hexo_y = one_hot_encode(val_hexo_y)
ohe_test_hexo_y = one_hot_encode(test_hexo_y)

# Run broad grid search to determine hyperparameters to center search
# start = time.time()
# hp_grid = list(it.product(FIRST_NNEURONS_START, SECOND_NNEURONS_START, LR_START, DROPOUT_START, BATCH_NORM_START,
#                           BATCH_SIZE_START, NUM_EPOCHS_START))
# hp_results = pd.DataFrame(columns=['first_nneurons', 'second_nneurons', 'act_fcn', 'lr', 'dropout', 'batch_norm',
#                                    'batch_size', 'num_epochs', 'acc', 'out_percent', 'sleep_percent', 'home_percent'])
# for first_nneurons, second_nneurons, lr, dropout, batch_norm, batch_size, num_epochs in hp_grid:
#     hexo_ann = create_ann(first_nneurons, second_nneurons, 'sigmoid', lr, dropout, batch_norm)
#     ann_fit = hexo_ann.fit(x=norm_train_hexo_x,
#                            y=ohe_train_hexo_y,
#                            batch_size=batch_size,
#                            epochs=num_epochs,
#                            validation_data=(norm_val_hexo_x, ohe_val_hexo_y),
#                            class_weight={0: w0, 1: w1, 2: w2})
#     val_pred = np.array([np.argmax(c) for c in hexo_ann.predict(norm_val_hexo_x)], dtype=float)
#     acc, out_percent, home_percent, sleep_percent = calc_metrics(val_pred, val_hexo_y)
#     hp_results = hp_results.append(pd.Series([first_nneurons, second_nneurons, 'sigmoid', lr, dropout, batch_norm,
#                                              batch_size, num_epochs, acc, out_percent, home_percent, sleep_percent],
#                                              index=hp_results.columns),
#                                    ignore_index=True)
# opt_ind = np.argmax(hp_results['acc'])
# opt_param = hp_results.iloc[opt_ind][['first_nneurons', 'second_nneurons', 'act_fcn', 'lr', 'dropout', 'batch_norm',
#                                       'batch_size', 'num_epochs']]
# opt_acc = hp_results.iloc[opt_ind]['acc']
# grid_search_time = time.time() - start
# hp_results.to_csv('results/ann/hexoskin_ann-broad-hp-results.csv', index=False)

# PARAMETER VALUES TO CENTER SEARCH
# first_nneurons: 3
# second_nneurons: 3
# lr: 0.01
# dropout: 0
# batch_norm: True
# batch_size: 5
# num_epochs: 100
# OPTIMAL VALIDATION PERFORMANCE
# acc: 91.5%
# out_percent: 100%
# home_percent: 81.0%
# sleep_percent: 81.0%
# SEARCH TIME: 2.7h

# Run focused grid search to optimize ANN parameters
# first_nneurons_g = [2, 3, 4]
# second_nneurons_g = [2, 3, 4]
# act_fcn_g = ['tanh', 'sigmoid']
# lr_g = [1e-3, 1e-2, 1e-1]
# dropout_g = [0, 0.125]
# batch_norm_g = [True, False]
# batch_size_g = [5, 47]
# num_epochs_g = [75, 100, 125, 150, 175]

# Run grid search to optimize ANN parameters
# start = time.time()
# hp_grid = list(it.product(first_nneurons_g, second_nneurons_g, act_fcn_g, lr_g, dropout_g, batch_norm_g, batch_size_g,
#                           num_epochs_g))
# hp_results = pd.DataFrame(columns=['first_nneurons', 'second_nneurons', 'act_fcn', 'lr', 'dropout', 'batch_norm',
#                                    'batch_size', 'num_epochs', 'acc', 'out_percent', 'home_percent', 'sleep_percent'])
# for first_nneurons, second_nneurons, act_fcn, lr, dropout, batch_norm, batch_size, num_epochs in hp_grid:
#     hexo_ann = create_ann(first_nneurons, second_nneurons, act_fcn, lr, dropout, batch_norm)
#     ann_fit = hexo_ann.fit(x=norm_train_hexo_x,
#                            y=ohe_train_hexo_y,
#                            batch_size=batch_size,
#                            epochs=num_epochs,
#                            validation_data=(norm_val_hexo_x, ohe_val_hexo_y),
#                            class_weight={0: w0, 1: w1, 2: w2})
#     val_pred = np.array([np.argmax(c) for c in hexo_ann.predict(norm_val_hexo_x)], dtype=float)
#     acc, out_percent, home_percent, sleep_percent = calc_metrics(val_pred, val_hexo_y)
#     hp_results = hp_results.append(pd.Series([first_nneurons, second_nneurons, act_fcn, lr, dropout, batch_norm,
#                                              batch_size, num_epochs, acc, out_percent, home_percent, sleep_percent],
#                                              index=hp_results.columns),
#                                    ignore_index=True)
# opt_ind = np.argmax(hp_results['acc'])
# opt_param = hp_results.iloc[opt_ind][['first_nneurons', 'second_nneurons', 'act_fcn', 'lr', 'dropout', 'batch_norm',
#                                       'batch_size', 'num_epochs']]
# opt_acc = hp_results.iloc[opt_ind]['acc']
# grid_search_time = time.time() - start
# hp_results.to_csv('results/ann/hexoskin_ann-focused-hp-results.csv', index=False)

# RESULTS:
# OPTIMAL PARAMETERS
# first_nneurons: 3
# second_nneurons: 2
# act_fcn: 'sigmoid'
# lr: 0.1
# dropout: 0
# batch_norm: False
# batch_size: 5
# num_epochs: 125
# OPTIMAL VALIDATION PERFORMANCE
# acc: 93.6%
# out_percent: 100%
# home_percent: 85.7%
# sleep_percent: 100%
# SEARCH TIME: 3.4h

# Fit model on the training set, using the validation set to get best model, and get performance on the test set
chk_cbk = tf.keras.callbacks.ModelCheckpoint('results/ann/hexoskin_ann.h5',  # Model checkpoint callback
                                             monitor='val_loss',
                                             verbose=True,
                                             save_best_only=True,
                                             mode='min')
hexo_ann = create_ann(3, 2, 'sigmoid', 0.1, 0, False)
ann_fit = hexo_ann.fit(x=norm_train_hexo_x,
                       y=ohe_train_hexo_y,
                       batch_size=5,
                       epochs=125,
                       validation_data=(norm_val_hexo_x, ohe_val_hexo_y),
                       class_weight={0: w0, 1: w1, 2: w2},
                       callbacks=[chk_cbk])
val_curve = pd.DataFrame(ann_fit.history['val_loss'])
val_curve.to_csv('results/ann/hexoskin_val_curve.csv')
hexo_ann = tf.keras.models.load_model('results/ann/hexoskin_ann.h5')
test_pred = np.array([np.argmax(c) for c in hexo_ann.predict(norm_test_hexo_x)], dtype=float)
hexo_acc, hexo_out_percent, hexo_home_percent, hexo_sleep_percent = calc_metrics(test_pred, test_hexo_y)
conf_matrix = pd.crosstab(test_pred, test_hexo_y)
conf_matrix = conf_matrix.rename(index={0: 'P Out', 1: 'P Home', 2: 'P Sleep'},
                                 columns={0: 'A Out', 1: 'A Home', 2: 'A Sleep'})
conf_matrix.to_csv('results/ann/hexoskin_ann-conf-matrix.csv')

# RESULTS:
# acc: 87.8%
# out_percent: 87.5%
# home_percent: 76.2%
# sleep_percent: 100%

print('Done')
