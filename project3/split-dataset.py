import pandas as pd
import numpy as np

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 0


def split_dataset(data, name):
    # Determine class distribution to sample points evenly
    num_out = len(np.where(data['activity'] == 0)[0])
    num_home = len(np.where(data['activity'] == 1)[0])
    num_sleep = len(np.where(data['activity'] == 2)[0])

    # Separate data points into classes
    out_pts = data.loc[data['activity'] == 0].reset_index(drop=True)
    home_pts = data.loc[data['activity'] == 1].reset_index(drop=True)
    sleep_pts = data.loc[data['activity'] == 2].reset_index(drop=True)

    # Get training set
    train_out_ind = np.ravel(pd.DataFrame(np.arange(num_out)).sample(n=int(np.round(num_out * TRAIN_RATIO)),
                                                                     random_state=SEED))
    train_home_ind = np.ravel(pd.DataFrame(np.arange(num_home)).sample(n=int(np.round(num_home * TRAIN_RATIO)),
                                                                       random_state=SEED))
    train_sleep_ind = np.ravel(pd.DataFrame(np.arange(num_sleep)).sample(n=int(np.round(num_sleep * TRAIN_RATIO)),
                                                                         random_state=SEED))
    train_out = out_pts.iloc[train_out_ind]
    train_home = home_pts.iloc[train_home_ind]
    train_sleep = sleep_pts.iloc[train_sleep_ind]
    train_data = pd.concat((train_out, train_home, train_sleep), axis=0)
    train_data.to_csv('train-' + name + '.csv', index=False)

    # Get test set
    test_val_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    tv_out_ind = pd.DataFrame([c for c in np.arange(num_out) if c not in train_out_ind])
    tv_home_ind = pd.DataFrame([c for c in np.arange(num_home) if c not in train_home_ind])
    tv_sleep_ind = pd.DataFrame([c for c in np.arange(num_sleep) if c not in train_sleep_ind])
    test_out_ind = np.ravel(pd.DataFrame(tv_out_ind).sample(n=int(np.ceil(len(tv_out_ind) * test_val_ratio)),
                                                            random_state=SEED))
    test_home_ind = np.ravel(pd.DataFrame(tv_home_ind).sample(n=int(np.ceil(len(tv_home_ind) * test_val_ratio)),
                                                              random_state=SEED))
    test_sleep_ind = np.ravel(pd.DataFrame(tv_sleep_ind).sample(n=int(np.ceil(len(tv_sleep_ind) * test_val_ratio)),
                                                                random_state=SEED))
    test_out = out_pts.iloc[test_out_ind]
    test_home = home_pts.iloc[test_home_ind]
    test_sleep = sleep_pts.iloc[test_sleep_ind]
    test_data = pd.concat((test_out, test_home, test_sleep), axis=0)
    test_data.to_csv('test-' + name + '.csv', index=False)

    # Get validation set
    val_out_ind = np.ravel(pd.DataFrame([c for c in tv_out_ind.values if c not in test_out_ind]))
    val_home_ind = np.ravel(pd.DataFrame([c for c in tv_home_ind.values if c not in test_home_ind]))
    val_sleep_ind = np.ravel(pd.DataFrame([c for c in tv_sleep_ind.values if c not in test_sleep_ind]))
    val_out = out_pts.iloc[val_out_ind]
    val_home = home_pts.iloc[val_home_ind]
    val_sleep = sleep_pts.iloc[val_sleep_ind]
    val_data = pd.concat((val_out, val_home, val_sleep), axis=0)
    val_data.to_csv('val-' + name + '.csv', index=False)


# Empatica data set
empatica = pd.read_csv('empatica.csv')
split_dataset(empatica, 'empatica')

# Hexoskin data set
hexoskin = pd.read_csv('hexoskin.csv')
split_dataset(hexoskin, 'hexoskin')
print('Done')
