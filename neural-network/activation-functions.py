import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'font.size': 32,
            'font.sans-serif': 'Verdana',
            'axes.titlesize': 36,
            'axes.labelsize': 36,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'legend.fontsize': 32,
            'legend.title_fontsize': 32,
            'legend.loc': 'center right',
            'legend.markerscale': 3.0,
            'figure.figsize': (24, 13.5)},
        style='whitegrid')

x = np.arange(-10, 10, step=0.01)

# Sigmoid activation function
y = 1/(1+np.exp(-x))
fig = plt.figure(figsize=(10, 10))
plt.plot(x, y, linewidth=4)
plt.savefig('sigmoid.png', bbox_inches='tight')
plt.close()

# Hyperbolic tangent function
y = np.tanh(x)
fig = plt.figure(figsize=(10, 10))
plt.plot(x, y, linewidth=4)
plt.savefig('tanh.png', bbox_inches='tight')
plt.close()

# Rectified linear unit (ReLU) function
y = np.arange(-10, 10, step=0.01)
y[np.where(y <= 0)] = 0
fig = plt.figure(figsize=(10, 10))
plt.plot(x, y, linewidth=4)
plt.savefig('relu.png', bbox_inches='tight')
plt.close()

# Linear function
fig = plt.figure(figsize=(10, 10))
plt.plot(x, x, linewidth=4)
plt.savefig('linear.png', bbox_inches='tight')
plt.close()
