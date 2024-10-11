"""
This script uses selection/reduction methods to select/reduce indicators on all datasets.
"""

import sys
sys.path.append('.')

import os
import time
import numpy as np
import pandas as pd
from miniutils import *
from issdv2 import issd
import time
# ISSD
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
# ECS/ECP
from baselines.ChannelSelectionMTSC.src.classelbow import ElbowPair # ECP
from baselines.ChannelSelectionMTSC.src.elbow import elbow # ECS
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# SFM
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
# umap and pca
import umap
from sklearn.decomposition import PCA
# surpress warnings, which are harmless
import warnings
warnings.filterwarnings("ignore")

def run(method, dataset):
    n_components = 4
    raw_data_path = f'data/{dataset}/raw'
    data_output_path = f'data/{dataset}/{method}' # path to save the selected data
    os.makedirs(data_output_path, exist_ok=True)

    if method not in ['issd', 'mi', 'ecs', 'ecp', 'lda', 'sfm', 'pca', 'umap']:
        raise ValueError(f'Unsupported method: {method}')

    fname_list = os.listdir(raw_data_path)
    fname_list.sort()

    if method == 'issd':
        time_start = time.time()
        # precompute statistics for all time series
        for fname in fname_list:
            data, label = load_data(os.path.join(raw_data_path, fname))
            result = issd(data,
                        label,
                        n_components,
                        strategy='qf',
                        n_jobs=20,
                        save_path=f'output/issd-cf/{dataset}/{fname[:-4]}')

        # devide the dataset into two parts
        part1_list = fname_list[:len(fname_list)//2]
        part2_list = fname_list[len(fname_list)//2:]
        
        for i in range(2):
            # rotate the dataset
            if i == 0:
                fname_list_train = part1_list
                fname_list_test = part2_list
            else:
                fname_list_train = part2_list
                fname_list_test = part1_list

            selected_channels_qf = inte_issd(dataset, n_components, fname_list_train, 'qf')
            selected_channels_cf = inte_issd(dataset, n_components, fname_list_train, 'cf')

            score_qf = 0
            score_cf = 0
            for fname in fname_list_train:
                data, state_seq = load_data(f'data/{dataset}/raw/{fname}')
                reduced_data_issd_qf = data[:,selected_channels_qf]
                lda_issd_qf = LinearDiscriminantAnalysis(n_components=1).fit_transform(reduced_data_issd_qf, state_seq)
                score_qf += np.sum(mutual_info_regression(lda_issd_qf, state_seq))
    
                data, state_seq = load_data(f'data/{dataset}/raw/{fname}')
                reduced_data_issd_cf = data[:,selected_channels_cf]
                lda_issd_cf = LinearDiscriminantAnalysis(n_components=1).fit_transform(reduced_data_issd_cf, state_seq)
                score_cf += np.sum(mutual_info_regression(lda_issd_cf, state_seq))

            s_qf = score_qf
            s_cf = score_cf
            if s_qf >= s_cf:
                selected_channels = selected_channels_qf
            else:
                selected_channels = selected_channels_cf
        time_end = time.time()

    elif method in ['lda', 'ecp', 'ecs', 'sfm', 'mi']:
        # devide the dataset into two parts
        part1_list = fname_list[:len(fname_list)//2]
        part2_list = fname_list[len(fname_list)//2:]
        
        time_start = time.time()
        for i in range(2):
            # rotate the dataset
            if i == 0:
                fname_list_train = part1_list
                fname_list_test = part2_list
            else:
                fname_list_train = part2_list
                fname_list_test = part1_list
            data_list = []
            state_seq_list = []
            for fn_test in fname_list_train:
                data, state_seq = load_data(os.path.join(raw_data_path, fn_test))
                data_list.append(data)
                state_seq_list.append(state_seq)
            data = np.vstack(data_list)
            state_seq = np.concatenate(state_seq_list)
            # SFM
            if method == 'sfm':
                estimator = LogisticRegression()
                sfm = SelectFromModel(estimator, max_features=n_components, threshold=-np.inf)
                # threshold=-np.inf means that the features are selected according to the importance of the feature
                # using default threshold may not select the desired number (<=) of features
                # please see the documentation from scikit-learn for more details.
                sfm.fit(data, state_seq)
                result = sfm.get_support(indices=True)
                result = [int(e) for e in result]
                # print(result[:n_components])
                for fn_test in fname_list_test:
                    data, state_seq = load_data(os.path.join(raw_data_path, fn_test))
                    data_reduced = data[:,result]
                    data_reduced = np.vstack((data_reduced.T, state_seq)).T
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
                for fn_test in fname_list_test:
                    data, state_seq = load_data(os.path.join(raw_data_path, fn_test))
                    data_reduced = data[:,result]
                    data_reduced = np.vstack((data_reduced.T, state_seq)).T
            # LDA
            elif method == 'lda':
                # n_components cannot be larger than min(n_features, n_classes - 1).
                n_components = min(n_components, len(np.unique(state_seq)) - 1)
                lda = LinearDiscriminantAnalysis(n_components=n_components)
                lda.fit(data, state_seq)
                for fn_test in fname_list_test:
                    data, state_seq = load_data(os.path.join(raw_data_path, fn_test))
                    data_reduced = lda.transform(data)
                    data_reduced = np.vstack((data_reduced.T, state_seq)).T
        time_end = time.time()

    # reduction methods
    elif method in ['pca', 'umap']:
        time_start = time.time()
        for fname in fname_list:
            # data = np.load(os.path.join(raw_data_path, fname), allow_pickle=True)
            data, _ = load_data(f'data/{dataset}/raw/{fname}')
            data_raw = data[:,:-1]
            label = data[:,-1]
            if method == 'pca':
                data_reduced = PCA(n_components=n_components).fit_transform(data_raw)
                data_reduced = np.vstack((data_reduced.T, label)).T
            elif method == 'umap':
                data_reduced = umap.UMAP(n_components=n_components).fit_transform(data_raw)
                data_reduced = np.vstack((data_reduced.T, label)).T
        time_end = time.time()
    print(f'{method}: {time_end-time_start} seconds')
    return time_end-time_start

methods = ['issd', 'pca', 'lda', 'umap', 'sfm', 'ecp', 'ecs']
datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'SynSeg', 'USC-HAD']

import json
import os
import glob
import shutil

for i in range(5):
    result_json = {}
    for m in methods:
        for d in datasets:
            run_time = run(m, d)
            result_json[f'{m}_{d}'] = run_time
    with open(f'output/time_consumption/data/comparison_execution{i}.json', 'w') as f:
        json.dump(result_json, f)
    # delete the output/issd-cf folder, it is not empty
    folder = 'output/issd-cf'
    # os.removedirs(folder, dir_fd=None)
    shutil.rmtree(folder, ignore_errors=True, onerror=None)