import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def convert_name(str):
    if str == 'time2state':
        return 'Time2State'
    elif str == 'ticc':
        return 'TICC'
    elif str == 'e2usd':
        return 'E2USD'
    elif str == 'autoplait':
        return 'AutoPlait'
    else:
        return str

os.makedirs('archive/figs', exist_ok=True)

for metric in ['nmi']:
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
    # remove the second column, which is the result of raw data
    table = np.delete(table, 1, axis=1)
    print(table.shape)
    table = table/100
    num_dmethods = len(dmethod_name_list)
    num_datasets = int(table.shape[0]/num_dmethods)

    print(dataset_names)
    for i, dataset in enumerate(dataset_names):
        # gen indices of downstream methods for the current dataset
        indices = [i+num_datasets*j for j in range(num_dmethods)]
        t = table[indices]
        # print(t.shape)

        # box plot
        plt.style.use('classic')
        plt.rcParams['pdf.fonttype'] = 42
        plt.figure(figsize=(4, 4))
        # plt.boxplot(avg_score_on_datasets.T, patch_artist=True, showfliers=False)
        # use bold line for all the components
        plt.boxplot(t.T,
            patch_artist=True,
            whis=[0,100],
            widths=0.5,
            showfliers=True,
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=1.5))
        # print(dmethod_name_list)
        dmethod_name_list = [convert_name(name) for name in dmethod_name_list]
        plt.ylim(-0.05,1)
        plt.xticks(range(1, num_dmethods+1), dmethod_name_list, fontsize=12)
        plt.title(f'{dataset}', fontsize=18)
        # plt.grid(axis='y', lw=2, linestyle='--', color='gray')
        # plt.ylim(0.35, 0.85)
        plt.ylabel('NMI', fontsize=18)
        plt.xlabel('Downstream Methods', fontsize=18)
        plt.tight_layout()
        # plt.savefig(f'archive/figs/resilience_{dataset}_{metric}.pdf', bbox_inches='tight')
        plt.savefig(f'archive/figs/resilience_{dataset}_{metric}.png', bbox_inches='tight')
        plt.close()

    # avg_score_on_datasets = []
    # for i in range(num_dmethods):
    #     method_idx = [num_datasets*i+j for j in range(num_datasets)]
    #     method_rows = table[method_idx]
    #     # print(method_idx)
    #     avg_score_on_datasets.append(np.mean(method_rows, axis=0))
    # avg_score_on_datasets = np.array(avg_score_on_datasets)
    # print(avg_score_on_datasets.shape)

    # box plot
    # plt.style.use('classic')
    # plt.rcParams['pdf.fonttype'] = 42
    # plt.figure(figsize=(4.3, 4))
    # # plt.boxplot(avg_score_on_datasets.T, patch_artist=True, showfliers=False)
    # # use bold line for all the components
    # plt.boxplot(avg_score_on_datasets.T,
    #     patch_artist=True,
    #     whis=[0,100],
    #     widths=0.5,
    #     showfliers=True,
    #     boxprops=dict(linewidth=1.5),
    #     whiskerprops=dict(linewidth=1.5),
    #     capprops=dict(linewidth=1.5),
    #     medianprops=dict(linewidth=1.5))
    # plt.ylim(-0.05,1)
    # plt.xticks(range(1, num_dmethods+1), dmethod_name_list, fontsize=12)
    # # plt.grid(axis='y', lw=2, linestyle='--', color='gray')
    # # plt.ylim(0.35, 0.85)
    # plt.title(f'All Datasets', fontsize=16)
    # plt.ylabel('NMI', fontsize=16)
    # plt.xlabel('Downstream Methods', fontsize=16)
    # plt.tight_layout()
    # # plt.savefig(f'archive/figs/resilience_all_{metric}.pdf', bbox_inches='tight')
    # plt.savefig(f'archive/figs/resilience_all_{metric}.png', bbox_inches='tight')