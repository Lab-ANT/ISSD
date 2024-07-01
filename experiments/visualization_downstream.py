from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import sys
sys.path.append('.')
from miniutils import reorder_label, load_data
import argparse

# receive args
parser = argparse.ArgumentParser()
parser.add_argument('downstream', type=str, help='The downstream method name')
parser.add_argument('dataset', type=str, help='The dataset name')
args = parser.parse_args()
dmethod = args.downstream
dataset = args.dataset

os.makedirs(f'output/visualization-downstream/{dataset}/{dmethod}', exist_ok=True)

methods = ['raw', 'issd', 'issd-qf', 'issd-cf', 'lda', 'sfm', 'ecp' ,'ecs', 'pca', 'umap']

fname_list = os.listdir(f'data/{dataset}/raw/')
for fname in fname_list:
    plt.style.use('classic')
    plt.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(nrows=len(methods), figsize=(5,len(methods)*0.85))
    for m in methods:
        if not os.path.exists(f'data/{dataset}/{m}/{fname}'):
                continue
        data, _ = load_data(f'data/{dataset}/{m}/{fname}')
        prediction = np.load(f'output/results/{dmethod}/{dataset}/{m}/{fname}')
        label = prediction[0]
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
    plt.savefig(f'output/visualization-downstream/{dataset}/{dmethod}/{fname[:-4]}.png')
    plt.savefig(f'output/visualization-downstream/{dataset}/{dmethod}/{fname[:-4]}.pdf')
    plt.close()