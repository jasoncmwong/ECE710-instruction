import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data set
heart = pd.read_csv('heart.csv')

# Balance dataset
num_neg = np.sum(heart['outcome'] == 0)
num_pos = np.sum(heart['outcome'] == 1)
num_remove = num_neg - num_pos
neg_ind = heart.index[heart['outcome'] == 0]
data_ind = np.ravel(pd.DataFrame(np.arange(len(neg_ind))).sample(n=num_remove, random_state=0))
heart = heart.drop(neg_ind[data_ind])

# Set input and output variables
x = heart[['chol', 'oldpeak']]
y = heart['outcome']

# Get training and test sets
data_ind = pd.DataFrame(np.arange(len(heart)))
test_ind = np.ravel(data_ind.sample(n=int(np.round(len(heart) * 0.2)), random_state=0))
train_ind = [c for c in np.arange(len(heart)) if c not in test_ind]
train_x, train_y = x.iloc[train_ind], y.iloc[train_ind]
test_x, test_y = x.iloc[test_ind], y.iloc[test_ind]

# Plot unnormalized data
test_pt = test_x.loc[125]
neighbor = train_x.iloc[141]
fig = plt.figure(figsize=(18, 12))
plt.scatter(train_x['oldpeak'], train_x['chol'], s=100, zorder=2)
plt.scatter(test_pt['oldpeak'], test_pt['chol'], c='red', s=100, zorder=2)
plt.plot([test_pt['oldpeak'], neighbor['oldpeak']], [test_pt['chol'], neighbor['chol']], c='k', linewidth=3, zorder=1)
plt.xlabel('$x_{1}$', fontsize=24)
plt.ylabel('$x_{2}$', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('raw-data.png', bbox_inches='tight')
plt.close()

# Normalize input features
mean = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x = (train_x-mean)/std

# Plot normalized data
test_pt = (test_x.loc[125]-mean)/std
neighbor = train_x.iloc[141]
fig = plt.figure(figsize=(18, 12))
plt.scatter(train_x['oldpeak'], train_x['chol'], s=100, zorder=2)
plt.scatter(test_pt['oldpeak'], test_pt['chol'], c='red', s=100, zorder=2)
plt.plot([test_pt['oldpeak'], neighbor['oldpeak']], [test_pt['chol'], neighbor['chol']], c='k', linewidth=3, zorder=1)
plt.xlabel('$x_{1}$', fontsize=24)
plt.ylabel('$x_{2}$', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('norm-data.png', bbox_inches='tight')
plt.close()
