import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

clf_name_list = ['ISSD', 'PCA', 'UMAP', 'ECP', 'ECS', 'LDA', 'SFM']
dmethod_names = ['Time2State', 'TICC', 'E2USD', 'AutoPlait']
dmethod_names_lower = ['time2state', 'ticc', 'e2usd', 'autoplait']
# dataset_names = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'Synthetic']

num_dmethod = len(dmethod_names)

os.makedirs('output/figs', exist_ok=True)

for metric in ['ari', 'purity', 'nmi']:
    with open(f'output/summary_{metric}.txt', 'r') as f:
        lines = f.readlines()[3:-1]
        # remove \n at the end of each line
        lines = [line[:-1] for line in lines]
        # delete space in each line
        lines = [line.replace(' ', '') for line in lines]
        # remove '|' at the beginin and end of each line
        lines = [line[1:-1] for line in lines]
        # split each line by '|'
        lines = [line.split('|') for line in lines]
        # convert each element to float, 'x' should be converted to 0
        table = [[0 if x == 'x' else float(x) for x in line[1:]] for line in lines]
        # dataset_names = [line[0] for line in lines]
        num_datasets = len(np.unique([line[0].split('/')[0] for line in lines]))
        dataset_names = [line[0].split('/')[0] for line in lines][:num_datasets]
        method_names = np.unique([line[0].split('/')[1] for line in lines])
        print(dataset_names, method_names)
        table = np.array(table)
    table = table/100
    num_datasets = int(table.shape[0]/num_dmethod)

    print(table)

    plt.style.use('classic')
    # boxplot
    plt.figure(figsize=(8, 4))
    plt.title(f'{metric.upper()} Boxplot', fontsize=20)
    plt.boxplot(table, patch_artist=True)
    plt.xticks(range(1, 8), clf_name_list)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'output/figs/boxplot_{metric}.png')