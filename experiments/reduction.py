"""
Created by Chengyu on 2023/11/15.
This script uses PCA and UMAP to reduce the dimension of all datasets.
The result of human selection is also enclosed.
Human selection may have more than 4 channels.
"""

import os
import numpy as np
from sklearn.decomposition import PCA
import tqdm
import umap

script_path = os.path.dirname(__file__)
n_components = 4
dataset_list = ['MoCap', 'SynSeg', 'PAMAP2', 'ActRecTut', 'USC-HAD']
methods = ['human', 'pca', 'umap']
dataset_list = ['SynSeg']

def human(dataset):
    if dataset == 'MoCap':
        channels = [38, 26, 55, 48]
    elif dataset == 'SynSeg':
        channels = [12,13,14,15]
    elif dataset == 'ActRecTut':
        channels = [0,1,2,3,4,5,6,7,8,9]
    elif dataset == 'PAMAP2':
        channels = [4,5,6,21,22,23,38,39,40]
    elif dataset == 'USC-HAD':
        channels = [0,1,2,3,4,5]
    return channels

for m in methods:
    for dataset in dataset_list:
        print(f'Processing dataset: {dataset}, using {m}')
        data_path = os.path.join(script_path, f'../data/{dataset}/raw')
        output_path = os.path.join(script_path, f'../data/{dataset}/{m}/')
        if not os.path.exists(data_path):
            print('Dataset not found.')
            continue
        os.makedirs(output_path, exist_ok=True)
        f_list = os.listdir(data_path)
        for fname in tqdm.tqdm(f_list):
            data = np.load(os.path.join(data_path, fname), allow_pickle=True)
            data_raw = data[:,:-1]
            label = data[:,-1] # pca and umap does not use label
            if m == 'pca':
                data_reduced = PCA(n_components=n_components).fit_transform(data_raw)
                data_reduced = np.vstack((data_reduced.T, label)).T
            elif m == 'umap':
                data_reduced = umap.UMAP(n_components=n_components).fit_transform(data_raw)
                data_reduced = np.vstack((data_reduced.T, label)).T
            elif m == 'human':
                selected_channels = human(dataset)
                data_reduced = data[:,selected_channels]
                data_reduced = np.vstack((data_reduced.T, label)).T
                print(data_reduced.shape, fname)
            np.save(os.path.join(output_path, fname), data_reduced)