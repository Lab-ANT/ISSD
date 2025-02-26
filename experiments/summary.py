"""
Created by Chengyu.
This script summarizes the results of the experiments.
The results of baselines and downstream tasks are saved in the output folder.
"""

import os
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, confusion_matrix
import prettytable as pt

methods = ['issd', 'issd-qf', 'issd-cf', 'raw', 'pca', 'umap', 'ecp', 'ecs', 'lda', 'sfm']
downstream_methods = ['time2state', 'ticc', 'autoplait', 'e2usd']
datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']

script_path = os.path.dirname(__file__)

def purity_score(y_true, y_pred):
    contingency_matrix = confusion_matrix(y_true, y_pred)
    purity = np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    return purity

def summary(metric):
    pretty_table = pt.PrettyTable()
    pretty_table.field_names = ['Dataset/DMethod'] + methods

    for dm in downstream_methods:
        for dataset in datasets:
            row = [f'{dataset}/{dm}']
            for method in methods:
                fname_list = os.listdir(os.path.join(script_path, f'../data/{dataset}/raw'))
                score_list = []
                result_path = os.path.join(script_path, f'../output/results/{dm}/{dataset}/{method}')
                if not os.path.exists(result_path):
                    row.append('x')
                    continue
                non_converge_flag = False
                non_converge_cnt = 0
                # print and do not end line
                print(f'{dataset}/{dm}/{method}', end=': ')
                for fn in fname_list:
                    if not os.path.exists(os.path.join(result_path, fn)):
                            # print(f'{dataset}/{dm}/{method}/{fn} not exists')
                            non_converge_flag = True
                            non_converge_cnt += 1
                            score_list.append(0)
                            continue
                    data = np.load(os.path.join(result_path, fn), allow_pickle=True)
                    if metric == 'nmi':
                        score = normalized_mutual_info_score(data[0].astype(int), data[1].astype(int))
                    elif metric == 'ari':
                        score = adjusted_mutual_info_score(data[0].astype(int), data[1].astype(int))
                    elif metric == 'purity':
                        score = purity_score(data[0].astype(int), data[1].astype(int))
                    score_list.append(score)
                if non_converge_flag:
                    print(f'{dataset}/{dm}/{method} has {non_converge_cnt} non-converge results')
                else:
                    print('\t')
                nmi_mean = np.mean(np.array(score_list)*100)
                row.append(f'{nmi_mean:.2f}')

            pretty_table.add_row(row)
    print(pretty_table)
    # create file and save the summary table
    with open(os.path.join(script_path, f'../output/summary_{metric}.txt'), 'w') as f:
        f.write(str(pretty_table))

for metric in ['nmi', 'ari', 'purity']:
    summary(metric)