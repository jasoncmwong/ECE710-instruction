import pandas as pd
import scipy.stats as sps
import numpy as np

# Preprocessing data file if pickle file does not exist
df = pd.read_csv('raw-data.csv')

# Convert class into number to obtain most occuring class in rolling window.
# df_labels = df[['Activity']].copy()
# df_labels['Activity'] = pd.factorize(df_labels['Activity'])[0]
# df_labels = df_labels.rolling(15, min_periods=1).apply(lambda x: sps.mode(x)[0])[14::15]

# Get mean values of rolling window.
# df_features = df.drop('Activity', axis=1)
# df_features = df_features.rolling(15, min_periods=1).mean()[14::15]

# Join columns and save to pickle file
# df_combined = pd.concat([df_features, df_labels], axis=1)

#data downsampling,
# df[['Empatica_HR']] = df[['Empatica_HR']].fillna(method='ffill')
# df = df.dropna(axis=0)
# one_hot = pd.get_dummies(df['Activity'])
# df = df.drop('Activity', axis=1)
# df = df.join(one_hot)

# #15 min downsampling with mean,
one_hot = pd.get_dummies(df['Activity'])
df = df.drop('Activity', axis=1)
df = df.join(one_hot)
df_split = np.array_split(df, 357)
for i in range(357):
    df_split[i] = pd.DataFrame(data=[df_split[i].mean(skipna=True, numeric_only=True)], index = df_split[i].index)
    df_split[i].drop(df_split[i].index[1:3601], 0, inplace=True)
df = pd.concat(df_split, ignore_index=True)
df = df.dropna(axis=0)
print('Done')
