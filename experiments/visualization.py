"""
Created by Chengyu.
This is a case study of ISSD and baselines on the MoCap dataset,
which intuitively shows the selection results of each method.
The output figure is saved in output/case-studies/
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import sys
sys.path.append('.')
from miniutils import compact, reorder_label

methods = ['raw', 'issd', 'lda', 'sfm', 'ecp' ,'ecs', 'pca', 'umap']
dataset = ['SynSeg', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD']

for d in dataset:
    fname_list = os.listdir(f'data/{d}/raw/')
    for fname in fname_list:
        os.makedirs(f'output/visualization/{d}', exist_ok=True)
        plt.style.use('classic')
        plt.rcParams['pdf.fonttype'] = 42
        fig, ax = plt.subplots(nrows=len(methods), figsize=(5,len(methods)*0.85))
        for m in methods:
            if not os.path.exists(f'data/{d}/{m}/{fname}'):
                    continue
            data = np.load(f'data/{d}/{m}/{fname}', allow_pickle=True)
            label = data[:,-1]
            data = data[:,:-1]
            label = reorder_label(label)
            data = StandardScaler().fit_transform(data)
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            ax[methods.index(m)].plot(data, lw=1)
            label = np.vstack([label, label])
            # ax[methods.index(m)].imshow(label, cmap='tab20', aspect='auto', alpha=0.4, interpolation='nearest')
            ax[methods.index(m)].imshow(label, cmap='tab20', aspect='auto', alpha=0.4)
            length = data.shape[0]
            ax[methods.index(m)].set_xlim(0, length)
            ax[methods.index(m)].set_yticks([])
            for tick in ax[methods.index(m)].xaxis.get_major_ticks():
                # tick.label.set_fontsize(6)
                tick.label1.set_fontsize(10)
            ax[methods.index(m)].set_ylim(0, 1)
            ax[methods.index(m)].set_ylabel(m.upper())
        plt.tight_layout()
        plt.savefig(f'output/visualization/{d}/{fname[:-4]}.png')
        plt.savefig(f'output/visualization/{d}/{fname[:-4]}.pdf')
        plt.close()