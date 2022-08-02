import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

# Generate data set
np.random.seed(0)
cov = [[0.3, 0.2], [0.2, 0.2]]
data = np.dot(np.linalg.cholesky(cov), np.random.standard_normal((2, 100))).T

# Fit PCA model
pca = PCA(n_components=2).fit(data)

# Plot first principal component
m1 = pca.components_[0][1]/pca.components_[0][0]
val = np.arange(-1.5, 1.51, step=0.01)
fig, ax = plt.subplots(figsize=(12, 12))
ax.axis('off')
plt.scatter(data[:, 0], data[:, 1], s=100, zorder=2)
for i in range(len(data)):
    proj = np.dot(np.dot(data[i, :], pca.components_[0])/np.dot(pca.components_[0], pca.components_[0]), pca.components_[0])
    plt.scatter(proj[0], proj[1], c='red', s=50, zorder=2)
    plt.plot([proj[0], data[i, 0]], [proj[1], data[i, 1]], c='red', linewidth=2, zorder=1)
plt.plot(val, m1*val, c='k', linewidth=4, zorder=1)
plt.xlabel('$x_{1}$', fontsize=24)
plt.ylabel('$x_{2}$', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.savefig('first-pca.png', bbox_inches='tight')
plt.close()

# Plot first principal component without projections
m1 = pca.components_[0][1]/pca.components_[0][0]
val = np.arange(-1.5, 1.51, step=0.01)
fig, ax = plt.subplots(figsize=(12, 12))
ax.axis('off')
plt.scatter(data[:, 0], data[:, 1], s=100, zorder=2)
for i in range(len(data)):
    proj = np.dot(np.dot(data[i, :], pca.components_[0])/np.dot(pca.components_[0], pca.components_[0]), pca.components_[0])
    plt.scatter(proj[0], proj[1], c='red', s=50, zorder=2)
plt.plot(val, m1*val, c='k', linewidth=4, zorder=1)
plt.xlabel('$x_{1}$', fontsize=24)
plt.ylabel('$x_{2}$', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.savefig('first-pca-no-proj.png', bbox_inches='tight')
plt.close()

# Plot unoptimal first principal component
m_bad = -5
fig, ax = plt.subplots(figsize=(12, 12))
ax.axis('off')
plt.scatter(data[:, 0], data[:, 1], s=100, zorder=2)
for i in range(len(data)):
    proj = np.dot(np.dot(data[i, :], [1, m_bad])/np.dot([1, m_bad], [1, m_bad]), [1, m_bad])
    plt.scatter(proj[0], proj[1], c='red', s=50, zorder=2)
    plt.plot([proj[0], data[i, 0]], [proj[1], data[i, 1]], c='red', linewidth=2, zorder=1)
plt.plot(val, m_bad*val, c='k', linewidth=4, zorder=1)
plt.xlabel('$x_{1}$', fontsize=24)
plt.ylabel('$x_{2}$', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.savefig('pca-unoptimal.png', bbox_inches='tight')
plt.close()

# Plot second principal component
m2 = pca.components_[1][1]/pca.components_[1][0]
fig, ax = plt.subplots(figsize=(12, 12))
ax.axis('off')
plt.scatter(data[:, 0], data[:, 1], s=100, zorder=2)
for i in range(len(data)):
    proj = np.dot(np.dot(data[i, :], pca.components_[1])/np.dot(pca.components_[1], pca.components_[1]), pca.components_[1])
    plt.scatter(proj[0], proj[1], c='red', s=50, zorder=2)
    # plt.plot([proj[0], data[i, 0]], [proj[1], data[i, 1]], c='red', linewidth=2, zorder=1)
plt.plot(val, m2*val, c='k', linewidth=4, zorder=1)
plt.xlabel('$x_{1}$', fontsize=24)
plt.ylabel('$x_{2}$', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.savefig('second-pca.png', bbox_inches='tight')
plt.close()

# Plot original data
fig = plt.figure(figsize=(12, 12))
plt.scatter(data[:, 0], data[:, 1], s=100)
plt.xlabel('$x_{1}$', fontsize=24)
plt.ylabel('$x_{2}$', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.savefig('og-data.png', bbox_inches='tight')
plt.close()

# Plot transformed axes
pca_x = pca.transform(data)
fig = plt.figure(figsize=(12, 12))
plt.scatter(pca_x[:, 0], pca_x[:, 1], c='red', s=100)
plt.xlabel('$PC_{1}$', fontsize=24)
plt.ylabel('$PC_{2}$', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.savefig('pca-data.png', bbox_inches='tight')
plt.close()

# Get explained variance
exp_var = pca.explained_variance_

# Load mtcars data set to illustrate elbow method in more complex PCA
mtcars = pd.read_csv('mtcars.csv')
x = mtcars[['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec']]
x = (x-x.mean(axis=0))/x.std(axis=0)

# Display scree plot
fig = plt.figure(figsize=(18, 12))
pca = PCA(n_components=7).fit(x)
plt.plot(np.arange(1, 8), pca.explained_variance_, marker='o', markersize=15, mfc='red', mec='k', linewidth=3)
plt.xlabel('Number of Principal Components', fontsize=24)
plt.ylabel('Variation Explained', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('elbow-plot.png', bbox_inches='tight')
plt.close()
print('Done')
