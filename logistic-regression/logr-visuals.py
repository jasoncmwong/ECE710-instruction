import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# Plot natural logarithm from x of 0 to 1
x = np.arange(0.00001, 1, step=0.00001)
fig = plt.figure(figsize=(16, 12.5))
plt.plot(x, np.log(x), linewidth=4)
plt.xlim([-0.01, 1])
plt.xlabel('t', fontsize=24)
plt.ylabel('ln(t)', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('ln-prob.png', bbox_inches='tight')
plt.close()

# Load data set
default = pd.read_csv('default.csv')

# Set input and output variables
x = default['balance']
y = default['default']

# Get training and test sets
data_ind = pd.DataFrame(np.arange(len(default)))
test_ind = np.ravel(data_ind.sample(n=int(np.round(len(default) * 0.2)), random_state=0))
train_ind = [c for c in np.arange(len(default)) if c not in test_ind]
train_x, train_y = x[train_ind], y[train_ind]
test_x, test_y = x[test_ind], y[test_ind]

# Fit model and get equation
logr_model = sm.Logit(train_y, sm.add_constant(train_x)).fit()
print(logr_model.summary())

# Plot fit model
fig = plt.figure(figsize=(16, 12.5))
sns.regplot(x='balance', y='default', data=default.iloc[train_ind], logistic=True, ci=None, line_kws={'linewidth': 6})
plt.scatter(train_x, train_y, c='red', s=75)
plt.xlabel('Balance', fontsize=24)
plt.ylabel('Default', fontsize=24)
plt.savefig('slogr-fit.png', bbox_inches='tight')
plt.close()

# Calculate performance metrics
test_prob = logr_model.predict(sm.add_constant(test_x))
pred = test_prob >= 0.5
tp = np.sum(np.logical_and(test_y == 1, pred == 1))
tn = np.sum(np.logical_and(test_y == 0, pred == 0))
fp = np.sum(np.logical_and(test_y == 0, pred == 1))
fn = np.sum(np.logical_and(test_y == 1, pred == 0))
acc = (tp + tn) / len(test_y)
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp)

# Get ROC points
roc_pts = []
prob = np.arange(0, 1.01, step=0.01)
for i in prob:
    pred = test_prob >= i
    tp = np.sum(np.logical_and(test_y == 1, pred == 1))
    tn = np.sum(np.logical_and(test_y == 0, pred == 0))
    fp = np.sum(np.logical_and(test_y == 0, pred == 1))
    fn = np.sum(np.logical_and(test_y == 1, pred == 0))
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    roc_pts.append([fpr, tpr])

    if i == 0.5:
        half_roc = [fpr, tpr]
roc_pts = np.array(roc_pts)

# Plot ROC curve
fig = plt.figure(figsize=(16, 12.5))
plt.plot(roc_pts[:, 0], roc_pts[:, 1], linewidth=5, label='Experimental')
plt.plot(np.arange(0, 1.01, step=0.01), np.arange(0, 1.01, step=0.01), 'r--', linewidth=5, label='Random')
plt.vlines(0, 0, 1, colors='k', linestyles='-.', linewidth=5)
plt.hlines(1, 0, 1, colors='k', linestyles='-.', linewidth=5, label='Ideal')
plt.xlabel('False Positive Rate or 1 - Specificity', fontsize=24)
plt.ylabel('True Positive Rate or Sensitivity', fontsize=24)
plt.legend(fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('roc-curve.png', bbox_inches='tight')
plt.close()

# Plot fit model with 0.5 threshold indicated
x_half = (np.log(0.5/0.5) - logr_model.params[0])/logr_model.params[1]
fig = plt.figure(figsize=(16, 12.5))
sns.regplot(x='balance', y='default',
            data=default.iloc[train_ind],
            logistic=True,
            ci=None,
            line_kws={'linewidth': 6, 'zorder': 1})
plt.scatter(train_x, train_y, c='red', s=75)
plt.scatter(x_half, 0.5, c='magenta', s=100, zorder=2)
plt.vlines(x_half, -0.05, 1.05, 'k', linestyles='dashed', linewidth=4)
plt.xlabel('Balance', fontsize=24)
plt.ylabel('Default', fontsize=24)
plt.ylim([-0.05, 1.05])
plt.savefig('slogr-fit-0.5thresh.png', bbox_inches='tight')
plt.close()

# Calculate performance metrics under a threshold determined by the class distribution
dist_prob = np.sum(train_y == 1) / len(train_y)
pred = test_prob >= dist_prob
tp = np.sum(np.logical_and(test_y == 1, pred == 1))
tn = np.sum(np.logical_and(test_y == 0, pred == 0))
fp = np.sum(np.logical_and(test_y == 0, pred == 1))
fn = np.sum(np.logical_and(test_y == 1, pred == 0))
acc = (tp + tn) / len(test_y)
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp)

# Plot fit model with threshold determined by the class distribution
x_bal = (np.log(dist_prob/(1-dist_prob)) - logr_model.params[0])/logr_model.params[1]
fig = plt.figure(figsize=(16, 12.5))
sns.regplot(x='balance', y='default',
            data=default.iloc[train_ind],
            logistic=True,
            ci=None,
            line_kws={'linewidth': 6, 'zorder': 1})
plt.scatter(train_x, train_y, c='red', s=75)
plt.scatter(x_bal, dist_prob, c='magenta', s=100, zorder=2)
plt.vlines(x_bal, -0.05, 1.05, 'k', linestyles='dashed', linewidth=4)
plt.xlabel('Balance', fontsize=24)
plt.ylabel('Default', fontsize=24)
plt.ylim([-0.05, 1.05])
plt.savefig('slogr-fit-bal-thresh.png', bbox_inches='tight')
plt.close()

# Determine optimal point on the ROC curve, maximizing sens*spec, and calculate performance metrics
opt_ind = np.argmax([(1-c[0])*c[1] for c in roc_pts])
opt_thresh = prob[opt_ind]
pred = test_prob >= opt_thresh
tp = np.sum(np.logical_and(test_y == 1, pred == 1))
tn = np.sum(np.logical_and(test_y == 0, pred == 0))
fp = np.sum(np.logical_and(test_y == 0, pred == 1))
fn = np.sum(np.logical_and(test_y == 1, pred == 0))
acc = (tp + tn) / len(test_y)
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp)

# Plot ROC curve with optimal threshold indicated
fig = plt.figure(figsize=(16, 12.5))
plt.plot(roc_pts[:, 0], roc_pts[:, 1], linewidth=5, label='Experimental', zorder=1)
plt.vlines(0, 0, 1, colors='k', linestyles='-.', linewidth=5)
plt.hlines(1, 0, 1, colors='k', linestyles='-.', linewidth=5, label='Ideal')
plt.scatter(roc_pts[opt_ind, 0], roc_pts[opt_ind, 1], s=200, c='red', zorder=2)
plt.xlabel('False Positive Rate or 1 - Specificity', fontsize=24)
plt.ylabel('True Positive Rate or Sensitivity', fontsize=24)
plt.legend(fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('opt-roc-curve.png', bbox_inches='tight')
plt.close()

# Plot fit model with threshold determined by the ROC curve
x_roc = (np.log(opt_thresh/(1-opt_thresh)) - logr_model.params[0])/logr_model.params[1]
fig = plt.figure(figsize=(16, 12.5))
sns.regplot(x='balance', y='default',
            data=default.iloc[train_ind],
            logistic=True,
            ci=None,
            line_kws={'linewidth': 6, 'zorder': 1})
plt.scatter(train_x, train_y, c='red', s=75)
plt.scatter(x_roc, opt_thresh, c='magenta', s=100, zorder=2)
plt.vlines(x_roc, -0.05, 1.05, 'k', linestyles='dashed', linewidth=4)
plt.xlabel('Balance', fontsize=24)
plt.ylabel('Default', fontsize=24)
plt.ylim([-0.05, 1.05])
plt.savefig('slogr-fit-roc-thresh.png', bbox_inches='tight')
plt.close()

# Plot fit model with weighted loss
loss_logr_model = LogisticRegression(class_weight='balanced').fit(np.array(train_x).reshape(-1, 1), train_y)
pred = loss_logr_model.predict_proba(np.array(test_x).reshape(-1, 1))[:, 1] >= 0.5
z = loss_logr_model.intercept_[0] + loss_logr_model.coef_[0][0]*train_x
logr_fcn = np.array(sorted(np.vstack((train_x, np.exp(z)/(1+np.exp(z)))).T, key=lambda l: l[1]))
x_bal = (np.log(0.5/(1-0.5)) - loss_logr_model.intercept_[0])/loss_logr_model.coef_[0][0]
tp = np.sum(np.logical_and(test_y == 1, pred == 1))
tn = np.sum(np.logical_and(test_y == 0, pred == 0))
fp = np.sum(np.logical_and(test_y == 0, pred == 1))
fn = np.sum(np.logical_and(test_y == 1, pred == 0))
acc = (tp + tn) / len(test_y)
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp)
fig = plt.figure(figsize=(16, 12.5))
plt.plot(logr_fcn[:, 0], logr_fcn[:, 1], linewidth=6, zorder=1)
plt.scatter(train_x, train_y, c='red', s=75)
plt.scatter(x_bal, 0.5, c='magenta', s=100, zorder=2)
plt.vlines(x_bal, -0.05, 1.05, 'k', linestyles='dashed', linewidth=4)
plt.xlabel('Balance', fontsize=24)
plt.ylabel('Default', fontsize=24)
plt.ylim([-0.05, 1.05])
plt.savefig('slogr-fit-weighted-loss.png', bbox_inches='tight')
plt.close()
print('Done')
