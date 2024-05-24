"""
Created by Chengyu on 2023/11/15.
This script uses selection methods to select indicators on all datasets.
"""

import sys
sys.path.append('.')

import os
import tqdm
import argparse
import numpy as np
import pandas as pd
from miniutils import *
# ISSD
from issd import issd
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
# SFS
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
# ECS/ECP
from baselines.ChannelSelectionMTSC.src.classelbow import ElbowPair # ECP
from baselines.ChannelSelectionMTSC.src.elbow import elbow # ECS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
# umap and pca
import umap
from sklearn.decomposition import PCA
# surpress warnings
import warnings
warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser()
argparser.add_argument('dname', type=str, help='dataset name')
argparser.add_argument('method', type=str, help='selection method')
argparser.add_argument('n_components', type=int, help='number of components')
args = argparser.parse_args()
dataset = args.dname
method = args.method
n_components = args.n_components

print(f'Processing {dataset} using {method}')

raw_data_path = f'data/{dataset}/raw'
data_output_path = f'data/{dataset}/{method}' # save the selected data
selection_output_path = f'output/selection/' # save the selected channels idx
os.makedirs(data_output_path, exist_ok=True)
os.makedirs(selection_output_path, exist_ok=True)

if dataset not in ['MoCap', 'SynSeg', 'ActRecTut', 'PAMAP2', 'USC-HAD']:
    raise ValueError(f'Unsupported dataset: {dataset}, the dataset should be [MoCap|SynSeg|ActRecTut|PAMAP2|USC-HAD]')
if method not in ['issd-qf', 'issd-cf', 'sfs', 'ecs', 'ecp', 'lda', 'sfm', 'pca', 'umap']:
    raise ValueError(f'Unsupported method: {method}, to use pca/umap/human, please use reduction.py')

fname_list = os.listdir(raw_data_path)
fname_list.sort()
selected_channels = []

if method == 'issd-qf':
    # run on each time series
    for fname in tqdm.tqdm(fname_list):
        data = np.load(os.path.join(raw_data_path, fname), allow_pickle=True)
        label = data[:,-1].astype(int)
        data = data[:,:-1]
        result = issd(data,
                    label,
                    n_components,
                    strategy='qf',
                    save_path=f'output/issd-qf/{dataset}/{fname[:-4]}')
    # integrate
    for fn_test in fname_list:
        data, state_seq = load_data(os.path.join(raw_data_path, fn_test))
        selected_channels = inte_issd(dataset, 4, fn_test, 'qf')
        print(selected_channels)
        data_reduced = data[:,selected_channels]
        data_reduced = np.vstack((data_reduced.T, state_seq)).T
        np.save(os.path.join(f'data/{dataset}/{method}', fn_test), data_reduced)
elif method == 'issd-cf':
    for fname in tqdm.tqdm(fname_list):
        data = np.load(os.path.join(raw_data_path, fname), allow_pickle=True)
        label = data[:,-1].astype(int)
        data = data[:,:-1]
        result = issd(data,
                    label,
                    n_components,
                    strategy='cf',
                    save_path=f'output/issd-cf/{dataset}/{fname[:-4]}')
    # integrate
    for fn_test in fname_list:
        data, state_seq = load_data(os.path.join(raw_data_path, fn_test))
        selected_channels = inte_issd(dataset, 4, fn_test, 'cf')
        print(selected_channels)
        data_reduced = data[:,selected_channels]
        data_reduced = np.vstack((data_reduced.T, state_seq)).T
        np.save(os.path.join(f'data/{dataset}/{method}', fn_test), data_reduced)
elif method == 'sfs':
    selected_channels_for_each_ts = []
    for fname in tqdm.tqdm(fname_list):
        data = np.load(os.path.join(raw_data_path, fname), allow_pickle=True)
        label = data[:,-1].astype(int)
        data = data[:,:-1]
        # PAMAP2 is too large for sfs, requires 5x downsampling
        if dataset == 'PAMAP2':
            data = data[::5]
            label = label[::5]
        num_channels = data.shape[1]
        # when n_components is less than half of the number of channels, use forward selection
        # if n_components <= int(num_channels/2):
        #     direction = 'forward'
        # else:
        #     direction = 'forward'
        direction = 'forward'
        knn = KNeighborsClassifier(n_neighbors=4)
        sfs = SequentialFeatureSelector(knn,
                                        n_features_to_select=n_components,
                                        n_jobs=10,)
        sfs.fit_transform(data, label)
        result = sfs.get_support(indices=True)
        result = [int(e) for e in result]
        selected_channels_for_each_ts.append(result)
    with open(os.path.join(selection_output_path, f'{dataset}_{method}.txt'), 'w') as f:
        for fname, result in zip(fname_list, selected_channels_for_each_ts):
            f.write(f'{fname} {result}\n')
    # integrate
    for fn_test in fname_list:
        selected_channels = inte_from_txt(dataset, method, fn_test, 4)
        print(selected_channels)
        data, state_seq = load_data(os.path.join(raw_data_path, fn_test))
        data_reduced = data[:,selected_channels]
        data_reduced = np.vstack((data_reduced.T, state_seq)).T
        np.save(os.path.join(f'data/{dataset}/{method}', fn_test), data_reduced)
elif method in ['lda', 'ecp', 'ecs', 'sfm']:
    for fname in tqdm.tqdm(fname_list):
        data_list = []
        state_seq_list = []
        for fn_test in fname_list:
            if fn_test == fname:
                continue
            data, state_seq = load_data(os.path.join(raw_data_path, fn_test))
            data_list.append(data)
            state_seq_list.append(state_seq)
        test_data, test_label = load_data(os.path.join(raw_data_path, fname))
        data = np.vstack(data_list)
        state_seq = np.concatenate(state_seq_list)
        # LDA
        if method == 'lda':
            # n_components cannot be larger than min(n_features, n_classes - 1).
            n_components = min(n_components, len(np.unique(state_seq)) - 1)
            lda = LinearDiscriminantAnalysis(n_components=n_components)
            lda.fit(data, state_seq)
            data_reduced = lda.transform(test_data)
            data_reduced = np.vstack((data_reduced.T, test_label)).T
        # ECP & ECS
        elif method == 'ecs' or method == 'ecp':
            center='mad' # options: mean, median
            if method == 'ecs':
                elb  = elbow(distance = 'eu', center=center) # Select elbow class sum
            elif method == 'ecp':
                elb = ElbowPair(distance = 'eu', center=center) # Selects elbow class Pair
            segments, label = adapt_for_clf(data, state_seq)
            elb.fit(pd.DataFrame(segments), compact(label))
            result = elb.relevant_dims[:n_components]
            data_reduced = test_data[:,result]
            data_reduced = np.vstack((data_reduced.T, test_label)).T
        elif method == 'sfm':
            estimator = LogisticRegression()
            sfm = SelectFromModel(estimator, max_features=4)
            sfm.fit(data, state_seq)
            result = sfm.get_support(indices=True)
            result = [int(e) for e in result]
            print(result)
            data_reduced = test_data[:,result]
            data_reduced = np.vstack((data_reduced.T, test_label)).T
        np.save(os.path.join(f'data/{dataset}/{method}', fname), data_reduced)

# reduction methods
elif method in ['pca', 'umap']:
    for fname in tqdm.tqdm(fname_list):
        data = np.load(os.path.join(raw_data_path, fname), allow_pickle=True)
        data_raw = data[:,:-1]
        label = data[:,-1]
        if method == 'pca':
            data_reduced = PCA(n_components=n_components).fit_transform(data_raw)
            data_reduced = np.vstack((data_reduced.T, label)).T
        elif method == 'umap':
            data_reduced = umap.UMAP(n_components=n_components).fit_transform(data_raw)
            data_reduced = np.vstack((data_reduced.T, label)).T
        np.save(os.path.join(f'data/{dataset}/{method}', fname), data_reduced)