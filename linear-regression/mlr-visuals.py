import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data set
mtcars = pd.read_csv('mtcars.csv')

# Set input and output variables
x = mtcars[['wt', 'hp']]
y = mtcars['mpg']

# Fit model and get equation + R squared value
mlr_model = sm.OLS(y, sm.add_constant(x)).fit()
b = mlr_model.params.values
r_sq = mlr_model.rsquared
corr = np.sqrt(r_sq)
print(mlr_model.summary())

# Plot hyperplane
wt, hp = np.meshgrid(np.linspace(min(mtcars['wt']), max(mtcars['wt'])),
                     np.linspace(min(mtcars['hp']), max(mtcars['hp'])))
pred = b[0] + b[1]*wt + b[2]*hp
fig = plt.figure(figsize=(24, 12.5))
ax = fig.gca(projection='3d')
ax.view_init(15, 135)
ax.plot_surface(wt, hp, pred, alpha=0.75)
ax.scatter(x['wt'], x['hp'], y, color='r', s=40)
for i in range(len(y)):
    ax.plot([x['wt'][i], x['wt'][i]], [x['hp'][i], x['hp'][i]], [y[i], mlr_model.fittedvalues[i]], 'k', linewidth=2)
ax.set_xlabel('Weight', fontsize=14)
ax.set_ylabel('Horsepower', fontsize=14)
ax.set_zlabel('MPG', fontsize=14)
plt.savefig('mlr-plot.png', bbox_inches='tight')
plt.close()
print('Done')
