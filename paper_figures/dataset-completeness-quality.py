import numpy as np
import matplotlib.pyplot as plt

datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']

# completeness = -np.load('completeness_analysis/MoCap_completeness.npy')
# quality = np.load('completeness_analysis/MoCap_quality.npy')

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(4, 3.3))
table = []
for d in datasets:
    completeness = np.load(f'completeness_analysis/{d}_completeness.npy').T
    # parts = ax.violinplot(completeness, showmeans=False, showmedians=True, showextrema=False)
    table.append(completeness.flatten())
parts = ax.violinplot(table, showmeans=False, showmedians=True, showextrema=False)
# violin setting
for pc in parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_edgecolor('skyblue')
    pc.set_alpha(1)
box = ax.boxplot(table, widths=0.25, patch_artist=True, whis=[0,95])
ax.set_xticks(np.arange(1, len(datasets)+1))
ax.set_xticklabels(datasets, rotation=15, fontsize=10)
ax.set_ylabel('Completeness')
plt.subplots_adjust(left=0.15)
plt.savefig('completeness_analysis/fig_completeness_box.png')
plt.savefig('completeness_analysis/fig_completeness_box.pdf')
plt.close()

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(4, 3.3))
table = []
for d in datasets:
    quality = np.load(f'completeness_analysis/{d}_quality.npy')
    table.append(quality.flatten())
parts = ax.violinplot(table, showmeans=False, showmedians=True, showextrema=False)
# violin setting
for pc in parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_edgecolor('skyblue')
    pc.set_alpha(1)
box = ax.boxplot(table, widths=0.25, patch_artist=True)
ax.set_xticks(np.arange(1, len(datasets)+1))
ax.set_xticklabels(datasets, rotation=15, fontsize=10)
ax.set_ylabel('Quality')
plt.subplots_adjust(left=0.15)
plt.savefig('completeness_analysis/fig_quality_box.png')
plt.savefig('completeness_analysis/fig_quality_box.pdf')
plt.close()