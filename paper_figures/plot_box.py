import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# clf_name_list = ['ISSD', 'PCA', 'UMAP', 'ECP', 'ECS', 'LDA', 'SFM']
dmethod_names = ['Time2State', 'TICC', 'E2USD', 'AutoPlait']
dmethod_names_lower = ['time2state', 'ticc', 'e2usd', 'autoplait']
# dataset_names = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'Synthetic']

num_dmethod = len(dmethod_names)

os.makedirs('output/figs', exist_ok=True)

for metric in ['ari', 'purity', 'nmi']:
    with open(f'output/summary_{metric}.txt', 'r') as f:
        # retrieve method names
        lines = f.readlines()
        method_name_list = lines[1].split('|')[2:-1]
        method_name_list = [name.strip().upper() for name in method_name_list]
    with open(f'output/summary_{metric}.txt', 'r') as f:
        # retrieve downstream method names
        lines = f.readlines()
        lines = lines[3:-1]
        name_rows = [line.split('|')[1].strip() for line in lines]
        name_rows = [name.split('/')[1] for name in name_rows]
        # remove duplicates and keep the order
        dmethod_name_list = list(dict.fromkeys(name_rows))
    with open(f'output/summary_{metric}.txt', 'r') as f:
        # retrieve dataset names
        lines = f.readlines()
        lines = lines[3:-1]
        name_rows = [line.split('|')[1].strip() for line in lines]
        name_rows = [name.split('/')[0] for name in name_rows]
        # remove duplicates and keep the order
        dataset_name_list = list(dict.fromkeys(name_rows))
    print(f'methods: {method_name_list}, downstream methods: {dmethod_name_list}, datasets: {dataset_name_list}')
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
    num_dmethods = len(dmethod_name_list)
    num_datasets = int(table.shape[0]/num_dmethods)

    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(4, 3.2))

    parts = ax.violinplot(table, showmeans=False, showmedians=True, showextrema=False)

    # violin setting
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_edgecolor('skyblue')
        pc.set_alpha(1)

    box = ax.boxplot(table, widths=0.25, patch_artist=True, showmeans=True)

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

    # ax.set_yticks(np.arange(0, 1, 0.2))
    ax.set_ylim(0, 1.05)
    # ax.set_xticks(range(len(clf_name_list)), clf_name_list, rotation=45)
    ax.set_xticklabels(method_name_list, rotation=45)
    ax.set_ylabel(f'{metric.upper()}', fontsize=15)
    ax.yaxis.labelpad = -1

    ax.grid(axis='y', lw=1, linestyle='--', color='gray')

    plt.tight_layout()
    plt.savefig(f'output/figs/violinplot_{metric}.png')
    # plt.savefig(f'output/figs/violinplot_{metric}.pdf')