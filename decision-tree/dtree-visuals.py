import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

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

# Fit decision tree
dtree = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=7, random_state=0).fit(train_x, train_y)
fig = plt.figure(figsize=(24, 12.5))
tree.plot_tree(dtree,
               feature_names=x.columns,
               class_names=['Negative', 'Positive'],
               filled=True,
               impurity=False)
plt.savefig('dendrogram.png', bbox_inches='tight')
plt.close()

# Get feature importances
gini_importance = dtree.feature_importances_
print('Importances:\nnum_vess: {0:.2f}\nmax_hr: {1:.2f}\nchest_pain: {2:.2f}'.format(*gini_importance))

# Get performance metrics on test set
pred = dtree.predict(test_x)
tp = np.sum(np.logical_and(test_y == 1, pred == 1))
tn = np.sum(np.logical_and(test_y == 0, pred == 0))
fp = np.sum(np.logical_and(test_y == 0, pred == 1))
fn = np.sum(np.logical_and(test_y == 1, pred == 0))
acc = (tp + tn) / len(test_y)
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp)

# Set input and output variables for overfit model
x = heart[['num_vess', 'max_hr', 'chest_pain']]
y = heart['outcome']

# Get training and test sets for overfit model
data_ind = pd.DataFrame(np.arange(len(heart)))
test_ind = np.ravel(data_ind.sample(n=int(np.round(len(heart) * 0.2)), random_state=0))
train_ind = [c for c in np.arange(len(heart)) if c not in test_ind]
train_x, train_y = x.iloc[train_ind], y.iloc[train_ind]
test_x, test_y = x.iloc[test_ind], y.iloc[test_ind]

# Overfit model
ofit_dtree = DecisionTreeClassifier(random_state=0).fit(train_x, train_y)
fig = plt.figure(figsize=(24, 12.5))
tree.plot_tree(ofit_dtree,
               feature_names=x.columns,
               class_names=['Negative', 'Positive'],
               filled=True,
               impurity=False)
plt.savefig('ofit-dendrogram.png', bbox_inches='tight')
plt.close()

# Get performance metrics from overfit model on test set
pred = ofit_dtree.predict(test_x)
tp = np.sum(np.logical_and(test_y == 1, pred == 1))
tn = np.sum(np.logical_and(test_y == 0, pred == 0))
fp = np.sum(np.logical_and(test_y == 0, pred == 1))
fn = np.sum(np.logical_and(test_y == 1, pred == 0))
acc = (tp + tn) / len(test_y)
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp)
print(tp, tn, fp, fn)
print('Done')
