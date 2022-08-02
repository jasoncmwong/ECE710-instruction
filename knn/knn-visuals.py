import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

sns.set(rc={'font.size': 32,
            'font.sans-serif': 'Verdana',
            'axes.titlesize': 36,
            'axes.labelsize': 36,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'legend.fontsize': 32,
            'legend.title_fontsize': 32,
            'legend.loc': 'upper left',
            'legend.markerscale': 3.0,
            'figure.figsize': (24, 13.5)},
        style='whitegrid')

# Load data set
heart = pd.read_csv('sa-heart.csv')

# Balance dataset
num_neg = np.sum(heart['chd'] == 0)
num_pos = np.sum(heart['chd'] == 1)
num_remove = num_neg - num_pos
neg_ind = heart.index[heart['chd'] == 0]
data_ind = np.ravel(pd.DataFrame(np.arange(len(neg_ind))).sample(n=num_remove, random_state=0))
heart = heart.drop(neg_ind[data_ind])

# Set input and output variables
x = heart[['age', 'tobacco']]
y = heart['chd']

# Get training and test sets
data_ind = pd.DataFrame(np.arange(len(heart)))
test_ind = np.ravel(data_ind.sample(n=int(np.round(len(heart) * 0.2)), random_state=0))
train_ind = [c for c in np.arange(len(heart)) if c not in test_ind]
train_x, train_y = x.iloc[train_ind], y.iloc[train_ind]
test_x, test_y = x.iloc[test_ind], y.iloc[test_ind]

# Normalize input features
mean = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x = (train_x-mean)/std

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_x, train_y)
pred = knn.predict((test_x-mean)/std)
tp = np.sum(np.logical_and(test_y == 1, pred == 1))
tn = np.sum(np.logical_and(test_y == 0, pred == 0))
fp = np.sum(np.logical_and(test_y == 0, pred == 1))
fn = np.sum(np.logical_and(test_y == 1, pred == 0))
acc = (tp + tn) / len(test_y)
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp)

# Plot KNN model
test_pt = ((test_x-mean)/std).iloc[0]
nn_dist, nn_ind = knn.kneighbors((test_x-mean)/std)
sns.scatterplot(train_x['age'], train_x['tobacco'], hue=y, s=200)
plt.scatter(test_pt['age'], test_pt['tobacco'], c='red', s=200)
for i in range(nn_ind.shape[1]):
    plt.plot([train_x.iloc[nn_ind[0, i]]['age'], test_pt['age']],
             [train_x.iloc[nn_ind[0, i]]['tobacco'], test_pt['tobacco']], c='k')
plt.xlabel('Age')
plt.ylabel('Tobacco')
plt.legend(title='Outcome')
plt.savefig('knn-plot.png', bbox_inches='tight')
plt.close()
print('Done')
