import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


CLASS_DICT = {0: 'Out', 1: 'Home', 2: 'Sleep'}


def plot_class_dist(data, var_name, dataset):
    feature_lbl = data['activity'].unique()
    fig = plt.figure(figsize=(18, 9))
    for i in range(len(feature_lbl)):
        sns.kdeplot(data=data.loc[data['activity'] == feature_lbl[i]][var_name], label=CLASS_DICT[i], linewidth=3)
    plt.xlabel(var_name, fontsize=24)
    plt.ylabel('Density', fontsize=24)
    plt.title(dataset.capitalize(), fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('data-visualization/' + dataset + '-' + var_name + '.png', bbox_inches='tight')
    plt.close()


# Empatica data set
empatica = pd.read_csv('empatica.csv')

# Plot pairwise scatter plots and individual variable histograms
sns.pairplot(data=empatica)
plt.savefig('data-visualization/empatica-pairplot.png', bbox_inches='tight')
plt.close()

# Plot density plots for each individual variable, separated by classes
for var_name in [c for c in empatica.columns if c != 'activity']:
    plot_class_dist(empatica, var_name, 'empatica')

# Investigate correlation matrix to evaluate which features are highly correlated
# RESULTS: accy_var & hr_mean have highest absolute pairwise correlation (0.46) - no variables eliminated
empatica_x = empatica[[c for c in empatica.columns if c not in 'activity']]
plt.figure(figsize=(20, 20))
sns.heatmap(empatica_x.corr(), annot=True, cmap='hot', annot_kws={'fontsize': 20})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('data-visualization/empatica-corr-matrix.png', bbox_inches='tight')
plt.close()

# Hexoskin data set
hexoskin = pd.read_csv('hexoskin.csv')

# Plot pairwise scatter plots and individual variable histograms
sns.pairplot(data=hexoskin)
plt.savefig('data-visualization/hexoskin-pairplot.png', bbox_inches='tight')
plt.close()

# Plot density plots for each individual variable, separated by classes
for var_name in [c for c in hexoskin.columns if c != 'activity']:
    plot_class_dist(hexoskin, var_name, 'hexoskin')

# Investigate correlation matrix to evaluate which features are highly correlated
# RESULTS: rr_mean & hr_mean have highest absolute pairwise correlation (0.71) - no variables eliminated
hexoskin_x = hexoskin[[c for c in hexoskin.columns if c not in 'activity']]
plt.figure(figsize=(20, 20))
sns.heatmap(hexoskin_x.corr(), annot=True, cmap='hot', annot_kws={'fontsize': 20})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('data-visualization/hexoskin-corr-matrix.png', bbox_inches='tight')
plt.close()


print('Done')
