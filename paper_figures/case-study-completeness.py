import numpy as np
import matplotlib.pyplot as plt

datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']

for d in datasets:
    completeness = np.load(f'completeness_analysis/{d}_completeness.npy')
    quality = np.load(f'completeness_analysis/{d}_quality.npy')

    plt.style.use('classic')
    plt.figure(figsize=(10, 2))
    plt.imshow(completeness, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'completeness_analysis/fig_{d}_completeness.png')
    plt.close()

    plt.style.use('classic')
    plt.figure(figsize=(10, 2))
    plt.imshow(quality, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'completeness_analysis/fig_{d}_quality.png')
    plt.close()

completeness = np.load('completeness_analysis/MoCap_completeness.npy')
quality = np.load('completeness_analysis/MoCap_quality.npy')

plt.style.use('classic')
plt.figure(figsize=(10, 2))
plt.imshow(completeness, cmap='viridis', interpolation='nearest')
for pos, c in zip([58, 28, 38, 55], [100,200,300,400]):
    # add a red box around the corresponding column
    plt.plot([pos-0.5, pos-0.5], [-0.5, 3.5], 'r')
    plt.plot([pos+0.5, pos+0.5], [-0.5, 3.5], 'r')
    plt.plot([pos-0.5, pos+0.5], [-0.5, -0.5], 'r')
    plt.plot([pos-0.5, pos+0.5], [3.5, 3.5], 'r')

plt.colorbar()
# plt.xlim(0, 62)
# plt.ylim(0, 8)
plt.tight_layout()
plt.savefig('completeness_analysis/fig_completeness.png')
plt.close()

# plt.style.use('classic')
# plt.figure(figsize=(10, 2))
# plt.imshow(quality, cmap='viridis', interpolation='nearest')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('completeness_analysis/fig_quality.png')
# plt.close()