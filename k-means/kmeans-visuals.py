import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# Load data set
mall = pd.read_csv('mall.csv')
x = mall[['age', 'income']]

# Normalize data set
norm_x = (x-x.mean(axis=0))/x.std(axis=0)

# Find optimal k using elbow method
inertiae = []
for k in np.arange(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=0)
    pred = kmeans.fit_predict(norm_x)
    inertiae.append(kmeans.inertia_)

# Plot error vs. k to find optimal k
fig = plt.figure(figsize=(18, 12))
plt.plot(np.arange(2, 15), inertiae, marker='o', markersize=15, mfc='red', mec='k', linewidth=3)
plt.xlabel('k', fontsize=24)
plt.ylabel('Within-cluster Variation', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('elbow-plot.png', bbox_inches='tight')
plt.close()

# Optimal k is 5 - fit model again for visualization
kmeans = KMeans(n_clusters=5, random_state=0)
pred = kmeans.fit_predict(norm_x)

# Plot cluster visualization of final k-means model for normalized data
fig = plt.figure(figsize=(12, 12))
plt.scatter(norm_x['age'], norm_x['income'], c=pred, s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='r', marker='x', s=100, label='Cluster centroids')
plt.xlabel('Age', fontsize=24)
plt.ylabel('Income', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.savefig('kmeans-normalized.png', bbox_inches='tight')
plt.close()

# Plot cluster visualization of final k-means model for original data
fig = plt.figure(figsize=(12, 12))
plt.scatter(x['age'], x['income'], c=pred, s=100)
plt.xlabel('Age [years]', fontsize=24)
plt.ylabel('Income [k$]', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('kmeans-original.png', bbox_inches='tight')
plt.close()

# Plot visualization of clustering steps
# Centroid initialization
init_center = np.array([[-0.20402309, 1.00667355],
                        [-1.42100291, -0.55435578],
                        [1.29930492, -0.24976469],
                        [-0.70513242, -1.23968573],
                        [1.08454378, 1.53970795]])
fig = plt.figure(figsize=(12, 12))
plt.scatter(norm_x['age'], norm_x['income'], c='k', s=100)
plt.scatter(init_center[:, 0], init_center[:, 1], c='r', marker='x', s=100, label='Cluster centroids')
plt.xlabel('Age', fontsize=24)
plt.ylabel('Income', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.savefig('kmeans-start.png', bbox_inches='tight')
plt.close()

# Class assignment
dists = []
for pt in norm_x.values:
    dists.append([np.linalg.norm(pt-c) for c in init_center])
dists = np.array(dists)
pred = np.array([np.argmin(c) for c in dists])
fig = plt.figure(figsize=(12, 12))
plt.scatter(norm_x['age'], norm_x['income'], c=pred, s=100)
plt.scatter(init_center[:, 0], init_center[:, 1], c='r', marker='x', s=100, label='Cluster centroids')
plt.xlabel('Age', fontsize=24)
plt.ylabel('Income', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.savefig('kmeans-one-iter.png', bbox_inches='tight')
plt.close()

# Centroid calculation
centroids = []
for i in range(5):
    centroids.append(norm_x.iloc[np.where(pred == i)[0]].mean().values)
centroids = np.array(centroids)
fig = plt.figure(figsize=(12, 12))
plt.scatter(norm_x['age'], norm_x['income'], c=pred, s=100)
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='x', s=100, label='Cluster centroids')
plt.xlabel('Age', fontsize=24)
plt.ylabel('Income', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.savefig('kmeans-class-assign.png', bbox_inches='tight')
plt.close()
print('Done')
