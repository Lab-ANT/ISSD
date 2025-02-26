import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dmethod_names = ['Time2State', 'TICC', 'E2USD', 'AutoPlait']
# dmethod_names_lower = ['time2state', 'ticc', 'e2usd', 'autoplait']

# num_dmethod = len(dmethod_names)

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

    # partition the table by downstream methods
    avg_score_on_datasets = []
    for i in range(num_datasets):
        dataset_idx = [j*num_datasets+i for j in range(num_dmethods)]
        dataset_rows = table[dataset_idx]
        avg_score_on_datasets.append(np.mean(dataset_rows, axis=0))
    avg_score_on_datasets = np.array(avg_score_on_datasets)

    print(avg_score_on_datasets)

    plt.style.use('classic')
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(3*num_datasets, 3))
    for i, avg_scores in enumerate(avg_score_on_datasets):
        ax[i].bar(method_name_list,
                    avg_scores,
                    width=0.5,
                    color=['#c9393e', '#497fc0', '#29517c', '#9694e7', '#ecd268', '#9dc37d', '#ddd2a4', '#00B4B7', '#008F9D', '#916142'])
        # rotate the x-axis label
        ax[i].tick_params(axis='x', rotation=45)
        # set x-axis label font size
        ax[i].tick_params(axis='x', labelsize=12)
        ax[i].grid(axis='y', lw=2, linestyle='--', color='gray')
        # put grid behind the bars
        # ax[i].set_axisbelow(True)
        ax[i].set_title(f'{dataset_names[i]}', fontsize=17)
        ax[i].set_ylim(0, 1.1)
        if metric == 'ari':
            ax[i].set_ylabel('ARI', fontsize=17)
        elif metric == 'purity':
            ax[i].set_ylabel('Purity', fontsize=17)
        elif metric == 'nmi':
            ax[i].set_ylabel('NMI', fontsize=17)
        ax[i].yaxis.labelpad = -1

        # # Count the best results obtained by each method from the table and add a star to the top two ranked methods
        # # sort the average scores in descending order
        # sort_idx = np.argsort(avg_scores)[::-1]
        # # find the maximum value and its index
        # max_val = avg_scores[sort_idx[0]]
        # max_idx = sort_idx[0]
        # # find the second maximum value and its index
        # second_max_val = avg_scores[sort_idx[1]]
        # second_max_idx = sort_idx[1]
        # # set the y-axis limit
        # ax[i].set_ylim(0, max_val*1.2)
        # # add a red star to the best method
        # ax[i].scatter(max_idx, max_val+0.08, color='red', s=120, marker='*')
        # # add a blue star to the second best method
        # ax[i].scatter(second_max_idx, second_max_val+0.08, color='#5D8AA8', s=120, marker='*')
        
        # add starts to the best and second best
        # sort_idx = np.argsort(avg_scores)[::-1]
        # max_val = avg_scores[sort_idx[0]]
        # max_idx = sort_idx[0]
        # second_max_val = avg_scores[sort_idx[1]]
        # second_max_idx = sort_idx[1]
        # if max_val*1.1 > 100:
        #     ax[i].set_ylim(0, max_val*1.2)
        # ax[i].scatter(max_idx, max_val+0.08, color='red', s=120, marker='*')
        # ax[i].scatter(second_max_idx, second_max_val+0.08, color='#5D8AA8', s=120, marker='*')
        ax[i].set_xlim(-0.7, len(method_name_list)-0.3)
    plt.tight_layout()
    plt.savefig(f'output/figs/overall_performance_{metric}.png', bbox_inches='tight')
    # plt.savefig(f'output/figs/overall_performance_{metric}.pdf', bbox_inches='tight')
        
    