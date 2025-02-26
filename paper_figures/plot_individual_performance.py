import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dmethod_names = ['Time2State', 'TICC', 'E2USD', 'AutoPlait']
# dmethod_names_lower = ['time2state', 'ticc', 'e2usd', 'autoplait']

air_force_blue = '#5D8AA8'

for metric in ['ari', 'nmi', 'purity']:
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
        dataset_names = [line[0] for line in lines]
        table = np.array(table)

    num_dmethods = len(dmethod_name_list)
    num_datasets = len(dataset_name_list)
    # num_clf = table.shape[1]

    plt.style.use('classic')
    fig, ax = plt.subplots(nrows=num_dmethods, ncols=5, figsize=(3.5*num_datasets, 3.5*num_dmethods))
    for i in range(num_dmethods): # 4 downstream methods
        for j in range(num_datasets): # 5 datasets
            ax[i,j].bar(method_name_list,
                        table[i*5+j],
                        width=0.5,
                        color=['#c9393e', '#497fc0', '#29517c', '#9694e7', '#ecd268', '#9dc37d', '#ddd2a4'])
            info = dataset_names[i*5+j].split('/')
            mname = info[1]
            dname = info[0]
            # add bold title
            ax[i,j].set_title(f'{mname} on {dname}',
                            fontsize=19)
                            #   fontweight='bold')
            ax[i,j].set_ylim(0, 115)
            if metric == 'ari':
                ax[i,j].set_ylabel('ARI', fontsize=19)
            elif metric == 'nmi':
                ax[i,j].set_ylabel('NMI', fontsize=19)
            elif metric == 'purity':
                ax[i,j].set_ylabel('Purity', fontsize=19)
            # set ylabel closer to the plot
            ax[i,j].yaxis.labelpad = -10
            # rotate the x-axis label
            ax[i,j].tick_params(axis='x', rotation=45, labelsize=12)
            # set y tick font size
            ax[i,j].tick_params(axis='y', labelsize=14)
            # add horizontal grid, use -- to make the grid line more visible
            ax[i,j].grid(axis='y', lw=2, linestyle='--', color='gray')
            # set border width
            ax[i,j].spines['top'].set_linewidth(2)
            ax[i,j].spines['right'].set_linewidth(2)
            ax[i,j].spines['bottom'].set_linewidth(2)
            ax[i,j].spines['left'].set_linewidth(2)
            # ax[i,j].set_xlim([-0.5, 6.5])
            # add a yellow star to the highest value
            max_val = max(table[i*5+j])
            max_idx = np.argwhere(table[i*5+j]==max_val)[0][0]
            ax[i,j].scatter(max_idx, max_val+8, color='red', s=120, marker='*')
            # add a black start to the second highest value
            table[i*5+j][max_idx] = 0
            second_max_val = max(table[i*5+j])
            second_max_idx = np.argwhere(table[i*5+j]==second_max_val)[0][0]
            ax[i,j].scatter(second_max_idx, second_max_val+8, color=air_force_blue, s=120, marker='*')

    os.makedirs('output/figs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'output/figs/individual_performance_{metric}.png')
    plt.close()
