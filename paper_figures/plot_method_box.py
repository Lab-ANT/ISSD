import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

clf_name_list = ['ISSD', 'PCA', 'UMAP', 'ECP', 'ECS', 'LDA', 'SFM']
dmethod_names = ['Time2State', 'TICC', 'AutoPlait', 'E2USD']
dmethod_names_lower = ['time2state', 'ticc', 'autoplait', 'e2usd']
# dataset_names = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'Synthetic']

num_dmethod = len(dmethod_names)

os.makedirs('output/figs', exist_ok=True)

metric = 'nmi'

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

num_datasets = int(table.shape[0]/num_dmethod)

avg_score_on_datasets = []
for i in range(num_dmethod):
    method_idx = [num_datasets*i+j for j in range(num_datasets)]
    method_rows = table[method_idx]
    # print(method_idx)
    avg_score_on_datasets.append(np.mean(method_rows, axis=0))
avg_score_on_datasets = np.array(avg_score_on_datasets)/100
print(avg_score_on_datasets.shape)

# box plot
plt.style.use('classic')
plt.rcParams['pdf.fonttype'] = 42
plt.figure(figsize=(4.3, 4))
# plt.boxplot(avg_score_on_datasets.T, patch_artist=True, showfliers=False)
# use bold line for all the components
plt.boxplot(avg_score_on_datasets.T,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    medianprops=dict(linewidth=1.5))
plt.xticks(range(1, num_dmethod+1), dmethod_names, fontsize=12)
# plt.grid(axis='y', lw=2, linestyle='--', color='gray')
plt.ylim(0.35, 0.85)
plt.ylabel('NMI', fontsize=16)
plt.xlabel('Downstream Methods', fontsize=16)
plt.tight_layout()
plt.savefig(f'output/figs/box_{metric}.pdf', bbox_inches='tight')

# variance = np.var(avg_score_on_datasets, axis=1)
# variance = [np.max(avg_score_on_datasets[i]) - np.min(avg_score_on_datasets[i]) for i in range(num_dmethod)]
# variance = np.array(variance)
# print(variance.shape)
# plt.style.use('classic')
# plt.figure(figsize=(5, 3))
# plt.bar(dmethod_names, variance, color=['#c9393e', '#497fc0', '#29517c', '#9694e7'])
# plt.savefig(f'output/figs/variance_{metric}.png', bbox_inches='tight')