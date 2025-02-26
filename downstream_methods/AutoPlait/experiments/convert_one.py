"""
Created by Chengyu on 2024/2/24.
This script converts data to the format required by AutoPlait.
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('dataset', type=str, default='data')
argparser.add_argument('method', type=str, default='raw')

args = argparser.parse_args()
method = args.method
dataset = args.dataset

datasets = [dataset]
methods = [method]

for d in datasets:
    for m in methods:
        if not os.path.exists(f'data/{d}/{m}'):
            continue
        original_path = f'data/{d}/{m}'
        output_path = f'downstream_methods/AutoPlait/data/{d}/{m}'
        os.makedirs(output_path, exist_ok=True)
        fname_list = os.listdir(f'data/{d}/raw')
        fname_list.sort()
        num_cols_list = []
        for fname in fname_list:
            data = np.load(os.path.join(original_path, fname), allow_pickle=True)
            label = data[:,-1].astype(int)
            data = data[:,:-1]
            data = StandardScaler().fit_transform(data)
            np.savetxt(os.path.join(output_path, fname.replace('.npy', '.txt')),
                       data,
                       delimiter='\t',
                       fmt='%1.4f')
            num_cols_list.append(data.shape[1])
        fname_list = [f'data/{d}/{m}/{fname[:-4]}.txt\n' for fname in fname_list]
        with open(os.path.join(output_path, 'list'), 'w') as f:
            f.writelines(fname_list)
        with open(os.path.join(output_path, 'info'), 'w') as f:
            f.write('\n'.join([str(num_cols) for num_cols in num_cols_list]))