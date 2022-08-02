import pandas as pd
import numpy as np


def load_sets(name):
    # Load different sets
    train = pd.read_csv('train-' + name + '.csv')
    val = pd.read_csv('val-' + name + '.csv')
    test = pd.read_csv('test-' + name + '.csv')

    # Split into input and output
    input_col = [c for c in train.columns if c != 'activity']
    return train[input_col], train['activity'], val[input_col], val['activity'], test[input_col], test['activity']


def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    norm = (data-mean) / std
    return norm, mean, std


def calc_class_weights(targets):
    num_out = len(np.where(targets == 0)[0])
    num_home = len(np.where(targets == 1)[0])
    num_sleep = len(np.where(targets == 2)[0])

    w_out = len(targets) / (3 * num_out)
    w_home = len(targets) / (3 * num_home)
    w_sleep = len(targets) / (3 * num_sleep)
    return w_out, w_home, w_sleep


def calc_metrics(pred, targets):
    t_out = np.sum(np.logical_and(targets == 0, pred == 0))
    f_home_t_out = np.sum(np.logical_and(targets == 0, pred == 1))
    f_sleep_t_out = np.sum(np.logical_and(targets == 0, pred == 2))
    t_home = np.sum(np.logical_and(targets == 1, pred == 1))
    f_out_t_home = np.sum(np.logical_and(targets == 1, pred == 0))
    f_sleep_t_home = np.sum(np.logical_and(targets == 1, pred == 2))
    t_sleep = np.sum(np.logical_and(targets == 2, pred == 2))
    f_out_t_sleep = np.sum(np.logical_and(targets == 2, pred == 0))
    f_home_t_sleep = np.sum(np.logical_and(targets == 2, pred == 1))
    acc = (t_out + t_home + t_sleep) / len(targets)
    sens_out = t_out / (t_out + f_home_t_out + f_sleep_t_out)
    sens_home = t_home / (t_home + f_out_t_home + f_sleep_t_home)
    sens_sleep = t_sleep / (t_sleep + f_out_t_sleep + f_home_t_sleep)
    return acc, sens_out, sens_home, sens_sleep
