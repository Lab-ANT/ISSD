"""
Created by Chengyu.
Average the results of independent executions.
The execution are specified in line 61, use 1-5 as default.
"""

import numpy as np
import matplotlib.pyplot as plt
import prettytable as pt
import os

def read_summary(path):
    with open(path, 'r') as f:
        # retrieve method names
        lines = f.readlines()
        method_name_list = lines[1].split('|')[2:-1]
        method_name_list = [name.strip().upper() for name in method_name_list]
    with open(path, 'r') as f:
        # retrieve downstream method names
        lines = f.readlines()
        lines = lines[3:-1]
        name_rows = [line.split('|')[1].strip() for line in lines]
        name_rows = [name.split('/')[1] for name in name_rows]
        # remove duplicates and keep the order
        dmethod_name_list = list(dict.fromkeys(name_rows))
    with open(path, 'r') as f:
        # retrieve dataset names
        lines = f.readlines()
        lines = lines[3:-1]
        name_rows = [line.split('|')[1].strip() for line in lines]
        name_rows = [name.split('/')[0] for name in name_rows]
        # remove duplicates and keep the order
        dataset_name_list = list(dict.fromkeys(name_rows))
    print(f'methods: {method_name_list}, downstream methods: {dmethod_name_list}, datasets: {dataset_name_list}')
    with open(path, 'r') as f:
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
    # table = table/100
    table = table
    num_dmethods = len(dmethod_name_list)
    num_datasets = int(table.shape[0]/num_dmethods)
    return table, method_name_list, dmethod_name_list, dataset_name_list

for metric in ['ari', 'nmi', 'purity']:
    execution_list = []
    for execution in range(1, 6):
        table, method_name_list, dmethod_name_list, dataset_name_list = read_summary(f'archive/execution{execution}/summary_{metric}.txt')
        print(table)
        print(method_name_list)
        print(dmethod_name_list)
        print(dataset_name_list)
        execution_list.append(table)
    table = np.mean(np.array(execution_list), axis=0)
    # drop 3,4 columns
    table = np.delete(table, [1, 2], axis=1)
    # drop 3,4 in method_name_list
    method_name_list = np.delete(method_name_list, [1, 2]).tolist()

    print(table.shape)
    pretty_table = pt.PrettyTable()
    pretty_table.field_names = ['Dataset/DMethod'] + method_name_list
    num_dmethods = len(dmethod_name_list)
    num_datasets = len(dataset_name_list)
    for i, dwname in enumerate(dmethod_name_list):
        for j, dname in enumerate(dataset_name_list):
            row = [f'{dname}/{dmethod_name_list[i]}']
            content = [f'{table[i*num_datasets+j][k]:.2f}' for k in range(len(method_name_list))]
            row += content
            pretty_table.add_row(row)
    print(pretty_table)

    os.makedirs('output', exist_ok=True)
    with open(f'output/summary_{metric}.txt', 'w') as f:
        f.write(str(pretty_table))