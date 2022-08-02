import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load data set
mtcars = pd.read_csv('mtcars.csv')

# Set input and output variables
x = mtcars['wt']
y = mtcars['mpg']

# Fit model and get equation + R squared value
slr_model = sm.OLS(y, sm.add_constant(x)).fit()
b = slr_model.params.values
r_sq = slr_model.rsquared
corr = np.sqrt(r_sq)
print(slr_model.summary())

# Plot fitted linear regression
fig = plt.figure(figsize=(16, 12.5))
plt.plot(x, slr_model.fittedvalues, linewidth=4)
plt.scatter(x, y, c='red', s=100)
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y[i], slr_model.fittedvalues[i]], 'k', linewidth=3)
plt.savefig('slr-plot.png', bbox_inches='tight')
plt.close()
