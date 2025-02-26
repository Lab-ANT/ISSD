"""
Created by Chengyu.
plot the box plot of dataset completeness and quality.
"""

import numpy as np
import matplotlib.pyplot as plt

datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']

mocap = [0.9667, 0.9286, 0.9583, 1.0, 0.9653, 0.0, 0.6667, 1.0, 0.7667]
actrectut = [0.5986, 0.7004]
pamap2 = [0.7862, 0.3817, 0.8740, 0.8237, 0.4072, 0.8398, 0.7091, 0.5236]
uschad = [0.9811, 0.9962, 0.9886, 0.9848, 0.9962, 0.9697, 0.9848, 0.9621, 0.9962]
synseg = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
dataset_completeness_list = [mocap, actrectut, pamap2, uschad, synseg]

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(4, 3.3))
table = []
for d in datasets:
    completeness = np.load(f'archive/completeness_analysis/{d}_completeness.npy').T
    table.append(np.mean(completeness, axis=1))
parts = ax.violinplot(table, showmeans=False, showmedians=True, showextrema=False)
# violin setting
for pc in parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_edgecolor('skyblue')
    pc.set_alpha(1)
# box = ax.boxplot(table, widths=0.25, patch_artist=True, whis=[0,95])
# if want to set the marker of outliers, use the following code
box = ax.boxplot(table, widths=0.25, patch_artist=True, whis=[0,100],
                 flierprops=dict(marker='o', markersize=2, markerfacecolor='black'))

# plot the dataset completeness, each list in dataset_completeness_list should
# be scatter plot on the same x-axis
dataset_completeness_list = [np.mean(d) for d in dataset_completeness_list]
# print(dataset_completeness_list)

# print(table[0].shape)
max_list = [np.max(table[i]) for i in range(len(table))]
# print(max_list)

for i, d in enumerate(dataset_completeness_list):
    ax.scatter([i+1], d, color='red', s=18, marker='*')
# add legend for star
ax.scatter([], [], color='red', s=18, marker='*', label='Dataset Completeness')

ax.set_xticks(np.arange(1, len(datasets)+1))
ax.set_xticklabels(datasets, rotation=15, fontsize=10)
ax.set_ylabel('Completeness')
ax.set_ylim(-0.005, 1.005)
plt.subplots_adjust(left=0.15)
# legend, boxsize, fontsize, location
plt.legend(loc='upper left', fontsize=9)

plt.savefig('archive/figs/fig_completeness_box.png')
# plt.savefig('completeness_analysis/fig_completeness_box.pdf')
plt.close()

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(4, 3.3))
table = []
for d in datasets:
    quality = np.load(f'archive/completeness_analysis/{d}_quality.npy').T
    # table.append(quality.flatten())
    table.append(np.mean(quality, axis=1))
parts = ax.violinplot(table, showmeans=False, showmedians=True, showextrema=False)
# violin setting
for pc in parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_edgecolor('skyblue')
    pc.set_alpha(1)
box = ax.boxplot(table, widths=0.25, patch_artist=True, whis=[0,100])
ax.set_xticks(np.arange(1, len(datasets)+1))
ax.set_xticklabels(datasets, rotation=15, fontsize=10)
ax.set_ylabel('Quality')
ax.set_ylim(0,1)
plt.subplots_adjust(left=0.15)
plt.savefig('archive/figs/fig_quality_box.png')
# plt.savefig('archive/figs/fig_quality_box.pdf')
plt.close()