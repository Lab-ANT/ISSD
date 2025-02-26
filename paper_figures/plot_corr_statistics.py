"""
Created by Chengyu.
Statistic the correlation between channels in each dataset.
"""
import sys
sys.path.append('.')
import os
import pandas as pd
import numpy as np
from miniutils import load_data

os.makedirs('archive/figs', exist_ok=True)

datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'SynSeg', 'USC-HAD']

dataset_corr_list = []
for dataset in datasets:
    raw_data_path = f'data/{dataset}/raw'
    fname_list = os.listdir(raw_data_path)
    fname_list.sort()
    
    corr_matrix_list = []
    for fname in fname_list:
        data, _ = load_data(f'data/{dataset}/raw/{fname}')
        corr_matrix = pd.DataFrame(data).corr(method='pearson').to_numpy()
        corr_matrix[np.isnan(corr_matrix)] = 0 # two constant channels will yild nan
        # set diagonal to 0
        np.fill_diagonal(corr_matrix, 0)
        # print(corr_matrix)
        # get triangle matrix, exclude diagonal
        corr_matrix = np.triu(corr_matrix, k=1)
        corr_matrix_list.append(corr_matrix.flatten())
    avg_corr_matrix = np.mean(corr_matrix_list, axis=0)
    print(avg_corr_matrix.shape)
    dataset_corr_list.append(avg_corr_matrix)

import matplotlib.pyplot as plt

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(4, 3.3))
parts = ax.violinplot(dataset_corr_list, showmeans=False, showmedians=True, showextrema=False)
# violin setting
for pc in parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_edgecolor('skyblue')
    pc.set_alpha(1)
box = ax.boxplot(dataset_corr_list, widths=0.3, patch_artist=True, whis=[1,99], flierprops=dict(marker='o', markersize=2, markerfacecolor='#29517c', markeredgecolor='#29517c'))
ax.set_ylabel('Pearson Correlation')
# box setting
for patch in box['boxes']:
    patch.set(facecolor='white')
    # patch.set(edgecolor='black')

# mean setting
for mean in box['means']:
    mean.set(marker='^', markerfacecolor='green', markersize=5)

# median setting
for median in box['medians']:
    median.set(color='red')

# plt.xticks(range(1, len(datasets)+1), datasets, rotation=45)
# plt.xticks(range(1, len(datasets)+1), datasets, fontsize=9, fontweight='bold')
plt.xticks(range(1, len(datasets)+1), datasets, fontsize=11, rotation=15)
plt.tight_layout()
plt.savefig('archive/figs/corr_boxplot.png')
# plt.savefig('archive/figs/corr_boxplot.pdf')
plt.close()

