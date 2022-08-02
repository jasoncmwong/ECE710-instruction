import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd
import seaborn as sns

np.random.seed(0)

# Create linear and nonlinear data sets
x = 2*np.random.randn(100) + 5
err = 0.4*np.random.randn(100)
lin_y = 4 + 0.3*x + err
nlin_y = 7 + 0.3*x + 2*x**2 + x**3 + 20*np.random.randn(100)

# Fit linear regression models for both data sets
lin_lr = sm.OLS(lin_y, sm.add_constant(x)).fit()
nlin_lr = sm.OLS(nlin_y, sm.add_constant(x)).fit()

# Plot linear data set
fig = plt.figure(figsize=(16, 12.5))
plt.scatter(x, lin_y, c='red', s=100)
plt.plot(x, lin_lr.fittedvalues, linewidth=4)
plt.xlabel('x', fontsize=24)
plt.ylabel('y', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('linear-plot.png', bbox_inches='tight')
plt.close()

# Plot nonlinear data set
fig = plt.figure(figsize=(16, 12.5))
plt.scatter(x, nlin_y, c='red', s=100)
plt.plot(x, nlin_lr.fittedvalues, linewidth=4)
plt.xlabel('x', fontsize=24)
plt.ylabel('y', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('nonlinear-plot.png', bbox_inches='tight')
plt.close()

# Plot heteroscedastic data set
het_y = 7 + 0.5*x + err*x
het_lr = sm.OLS(het_y, sm.add_constant(x)).fit()
fig = plt.figure(figsize=(16, 12.5))
plt.scatter(x, het_y, c='red', s=100)
plt.plot(x, het_lr.fittedvalues, linewidth=4)
plt.xlabel('x', fontsize=24)
plt.ylabel('y', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('heteroscedastic-plot.png', bbox_inches='tight')
plt.close()

# Plot residuals vs. fitted values - linearity check
fig, ax = plt.subplots(figsize=(16, 12.5))
plt.scatter(lin_lr.fittedvalues, lin_lr.resid, c='red', s=100)
plt.xlabel('Fitted values', fontsize=24)
plt.ylabel('Residuals', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
xmin, xmax = ax.get_xlim()
plt.xlim([xmin, xmax])
plt.hlines(0, xmin, xmax, colors='k', linestyles='--', linewidth=4)
plt.savefig('linear-resid-fitted.png', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(16, 12.5))
plt.scatter(nlin_lr.fittedvalues, nlin_lr.resid, c='red', s=100)
plt.xlabel('Fitted values', fontsize=24)
plt.ylabel('Residuals', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
xmin, xmax = ax.get_xlim()
plt.xlim([xmin, xmax])
plt.hlines(0, xmin, xmax, colors='k', linestyles='--', linewidth=4)
plt.savefig('nonlinear-resid-fitted.png', bbox_inches='tight')
plt.close()

# Plot scale-location plot - homoscedasticity check
fig, ax = plt.subplots(figsize=(16, 12.5))
plt.scatter(lin_lr.fittedvalues, lin_lr.resid_pearson, c='red', s=100)
plt.xlabel('Fitted values', fontsize=24)
plt.ylabel('Square root of standardized residuals', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
xmin, xmax = ax.get_xlim()
plt.xlim([xmin, xmax])
plt.hlines(0, xmin, xmax, colors='k', linestyles='--', linewidth=4)
plt.savefig('homo-scale-location.png', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(16, 12.5))
plt.scatter(het_lr.fittedvalues, het_lr.resid_pearson, c='red', s=100)
plt.xlabel('Fitted values', fontsize=24)
plt.ylabel('Square root of standardized residuals', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
xmin, xmax = ax.get_xlim()
plt.xlim([xmin, xmax])
plt.hlines(0, xmin, xmax, colors='k', linestyles='--', linewidth=4)
plt.savefig('hetero-scale-location.png', bbox_inches='tight')
plt.close()

# Plot QQ plot - multivariate normality check
fig = sm.qqplot(lin_lr.resid, dist=stats.t, fit=True, line='45', markersize=10)
fig.set_figwidth(16)
fig.set_figheight(12.5)
plt.xlabel('Theoretical quantiles', fontsize=24)
plt.ylabel('Sample quantiles', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('linear-qq.png', bbox_inches='tight')
plt.close()

fig = sm.qqplot(nlin_lr.resid, dist=stats.t, fit=True, line='45', markersize=10)
fig.set_figwidth(16)
fig.set_figheight(12.5)
plt.xlabel('Theoretical quantiles', fontsize=24)
plt.ylabel('Sample quantiles', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('nonlinear-qq.png', bbox_inches='tight')
plt.close()

# Load California housing data set
cali = pd.read_csv('housing.csv')
cali_x = cali[[c for c in cali.columns if c not in ('ocean_proximity', 'median_house_value')]]
plt.figure(figsize=(14, 12))
sns.heatmap(cali_x.corr(), annot=True, cmap='hot', annot_kws={'fontsize': 20})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('corr-matrix.png', bbox_inches='tight')
plt.close()

cali_x = cali[[c for c in cali.columns if c not in ('ocean_proximity', 'median_house_value', 'bedrooms')]]
plt.figure(figsize=(14, 12))
sns.heatmap(cali_x.corr(), annot=True, cmap='hot', annot_kws={'fontsize': 20})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('corr-matrix2.png', bbox_inches='tight')
plt.close()
print('Done')
