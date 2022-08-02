import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd
import seaborn as sns

np.random.seed(0)

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
pred = logr_model.predict(sm.add_constant(test_x))

# Plot log_odds vs. balance - linearity with log-odds check
log_odds = np.log(pred/(1-pred))
fig = plt.figure(figsize=(16, 12.5))
plt.scatter(test_x, log_odds, c='red', s=100)
plt.xlabel('Balance', fontsize=24)
plt.ylabel('Log-odds', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('linear-log-odds.png', bbox_inches='tight')
plt.close()
print('Done')
