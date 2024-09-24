import numpy as np
import matplotlib.pyplot as plt

datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']

# completeness = -np.load('completeness_analysis/MoCap_completeness.npy')
# quality = np.load('completeness_analysis/MoCap_quality.npy')

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(4, 3.2))
table = []
for d in datasets:
    completeness = np.load(f'completeness_analysis/{d}_completeness.npy').T
    # parts = ax.violinplot(completeness, showmeans=False, showmedians=True, showextrema=False)
    # parts = 
    table.append(completeness.flatten())
parts = ax.violinplot(table, showmeans=False, showmedians=True, showextrema=False)
# violin setting
for pc in parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_edgecolor('skyblue')
    pc.set_alpha(1)

box = ax.boxplot(table, widths=0.25, patch_artist=True, whis=[0,100],)
plt.savefig('completeness_analysis/fig_completeness_box.png')
plt.close()

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(4, 3.2))
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
plt.savefig('completeness_analysis/fig_quality_box.png')
plt.close()


# plt.style.use('classic')
# plt.figure(figsize=(10, 2))
# plt.imshow(completeness, cmap='viridis', interpolation='nearest')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('completeness_analysis/fig_completeness.png')
# plt.close()

# plt.style.use('classic')
# plt.figure(figsize=(10, 2))
# plt.imshow(quality, cmap='viridis', interpolation='nearest')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('completeness_analysis/fig_quality.png')
# plt.close()