import pandas as pd
import numpy as np

HR_RR_RATE = 4
ACC_RATE = 1


def downsample_data(raw_data, target_str, rate=3600, min_ratio=2/3):
    # Create data frame to store mean and variance of each input variable
    feature_lbl = [c for c in raw_data.columns if c != target_str]
    ds_data = pd.DataFrame(columns=[c for cc in [(c + '_mean', c + '_var') for c in feature_lbl] for c in cc] + ['activity'])

    # Remove initial and ending data points with all NaN feature values
    nan_ind = np.where(raw_data[feature_lbl].isnull().all(axis=1))[0]
    nan_groups = np.split(nan_ind, np.where(np.diff(nan_ind) > 1)[0] + 1)
    if len(ds_data)-1 in nan_groups[-1]:
        raw_data = raw_data.drop(index=nan_groups[-1])
    if 0 in nan_groups[0]:
        raw_data = raw_data.drop(index=nan_groups[0])

    # Iterate over different classes separately to downsample data
    class_lbl = raw_data[target_str].unique()
    for i in class_lbl:
        # Find consecutive groups of classes
        class_ind = np.where(raw_data[target_str] == i)[0]
        class_groups = np.split(class_ind, np.where(np.diff(class_ind) > 1)[0] + 1)

        # Iterate over consecutive groups and downsample within each group
        for group in class_groups:
            # Separate groups into clusters of the required downsampling rate ("rate" points -> 1 data point)
            num_pts = int(np.round(len(group) / rate))
            for period in np.arange(num_pts):
                # Get upper limit of range of points to consider for this group
                up_lim = group[0] + min((period+1)*rate, len(group))
                window = raw_data[feature_lbl].iloc[group[0]+period*rate:up_lim]

                # Check if window has enough valid data points to downsample
                val_count = np.sum(window[feature_lbl].notnull(), axis=0)
                if val_count['hr'] < min_ratio*rate/HR_RR_RATE or val_count['accx'] < min_ratio*rate/ACC_RATE or\
                   val_count['accy'] < min_ratio*rate/ACC_RATE or val_count['accz'] < min_ratio*rate/ACC_RATE or\
                   ('rr' in val_count and val_count['rr'] < min_ratio*rate/HR_RR_RATE):
                    continue  # Skip this window if not enough valid data points

                # Append to data set if there are enough valid points
                ds_data = ds_data.append(pd.Series([c for cc in zip(window.mean(), window.var()) for c in cc] + [i],
                                                   index=ds_data.columns),
                                         ignore_index=True)
    return ds_data


# Data set description:
# Two devices: Empatica wristband and Hexoskin torso
# Empatica measures heart rate (HR) and accelerometer data (ACCX, ACCY, ACCZ)
# Hexoskin measures heart rate (HR), respiratory rate (RR), and accelerometer data (ACCX, ACCY, ACCZ)
# All accelerometer data is sampled at 4Hz and all HR and RR data is sampled at 1Hz
# Desired prediction is "Activity", indicating whether the user is outside the home (OUT), sleeping/resting (SLEEP), or
# inside the house (HOME); in particular, trying to predict what activity a person is doing over a 15-minute period

# Load raw data set
data = pd.read_csv('raw-data.csv')

# Remove data points with NaN targets
nan_ind = np.where(data['Activity'].isnull().values)[0]
data = data.drop(index=nan_ind)

# Get expected number of data points for each class (assuming downsampling of 3600 points to 1)
num_out = len(np.where(data['Activity'] == 'Out')[0])/3600
num_home = len(np.where(data['Activity'] == 'Home')[0])/3600
num_sleep = len(np.where(data['Activity'] == 'Sleep')[0])/3600

# Encode activity as {'Out': 0, 'Home': 1, 'Sleep': 2}
data.loc[data['Activity'] == 'Out', 'Activity'] = 0
data.loc[data['Activity'] == 'Home', 'Activity'] = 1
data.loc[data['Activity'] == 'Sleep', 'Activity'] = 2

# Separate and re-name data sets
empatica = data[['Empatica_HR', 'Empatica_ACCX', 'Empatica_ACCY', 'Empatica_ACCZ', 'Activity']]
empatica = empatica.rename(columns={'Empatica_HR': 'hr',
                                    'Empatica_ACCX': 'accx',
                                    'Empatica_ACCY': 'accy',
                                    'Empatica_ACCZ': 'accz',
                                    'Activity': 'activity'})
hexoskin = data[['Hexoskin_HR', 'Hexoskin_RR', 'Hexoskin_ACCX', 'Hexoskin_ACCY', 'Hexoskin_ACCZ', 'Activity']]
hexoskin = hexoskin.rename(columns={'Hexoskin_HR': 'hr',
                                    'Hexoskin_RR': 'rr',
                                    'Hexoskin_ACCX': 'accx',
                                    'Hexoskin_ACCY': 'accy',
                                    'Hexoskin_ACCZ': 'accz',
                                    'Activity': 'activity'})

# Downsample data
empatica = downsample_data(empatica, 'activity')
hexoskin = downsample_data(hexoskin, 'activity')

# Save cleaned up data sets
empatica.to_csv('empatica.csv', index=False)
hexoskin.to_csv('hexoskin.csv', index=False)
print('Done')
