"""
Created by Chengyu on 2024/2/24.
This script converts data to the format required by AutoPlait.
"""

import os
import numpy as np
import re
import pandas as pd
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('dataset', type=str, default='data')
argparser.add_argument('method', type=str, default='raw')

args = argparser.parse_args()
method = args.method
dataset = args.dataset

datasets = [dataset]
methods = [method]

def len_of_file(path):
    return len(open(path,'r').readlines())

def read_result_dir(path,original_file_path):
    results = os.listdir(path)
    results = [s if re.match('segment.\d', s) != None else None for s in results]
    results = np.array(results)
    idx = np.argwhere(results!=None)
    results = results[idx].flatten()

    length = len_of_file(original_file_path)
    label = np.zeros(length,dtype=int)

    l = 0
    for r in results:
        data = pd.read_csv(path+r, names=['col0','col1'], header=None, index_col=False, sep = ' ')
        start = data.col0
        end = data.col1
        for s, e in zip(start,end):
            label[s:e+1]=l
        l+=1
    return label

for d in datasets:
    for m in methods:       
        original_path = f'data/{d}/{m}'
        result_path = f'downstream_methods/AutoPlait/output/{d}/{m}'
        if not os.path.exists(result_path):
            continue
        print(f'AutoPlait: Processing {m} on {d}...')
        temp_path = f'downstream_methods/AutoPlait/data/{d}/{m}'
        output_path = f'output/results/autoplait/{d}/{m}'
        os.makedirs(output_path, exist_ok=True)
        fname_list = os.listdir(f'data/{d}/raw')
        fname_list.sort()
        for i, fname in enumerate(fname_list):
            data = np.load(os.path.join(original_path, fname), allow_pickle=True)
            groundtruth = data[:,-1].astype(int)
            print(groundtruth.shape)
            prediction = read_result_dir(f'{result_path}/dat{i+1}/',
                                        f'{temp_path}/{fname.replace(".npy",".txt")}')
            np.save(os.path.join(output_path, fname), np.vstack((prediction, groundtruth)))