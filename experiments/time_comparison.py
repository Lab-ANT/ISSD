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
from issd import ISSD
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
        selector = ISSD(n_jobs=20)
        datalist = [load_data(os.path.join(raw_data_path, fname))[0] for fname in fname_list]
        state_seq_list = [load_data(os.path.join(raw_data_path, fname))[1] for fname in fname_list]

        selector.compute_matrices(datalist, state_seq_list)
        selected_channels_qf = selector.get_qf_solution(4)
        selected_channels_cf = selector.get_cf_solution(4)
        selected_channels = selector.inte_solution()
        time_end = time.time()

    elif method in ['lda', 'ecp', 'ecs', 'sfm', 'mi']:
        time_start = time.time()
        data_list = []
        state_seq_list = []
        for fn_test in fname_list:
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
            # result = sfm.get_support(indices=True)
            # result = [int(e) for e in result]
        # ECP & ECS
        elif method == 'ecs' or method == 'ecp':
            center='mad' # options: mean, median
            if method == 'ecs':
                elb  = elbow(distance = 'eu', center=center) # Select elbow class sum
            elif method == 'ecp':
                elb = ElbowPair(distance = 'eu', center=center) # Selects elbow class Pair
            segments, label = adapt_for_clf(data, state_seq)
            elb.fit(pd.DataFrame(segments), compact(label))
            # result = elb.relevant_dims[:n_components]
        # LDA
        elif method == 'lda':
            # n_components cannot be larger than min(n_features, n_classes - 1).
            n_components = min(n_components, len(np.unique(state_seq)) - 1)
            lda = LinearDiscriminantAnalysis(n_components=n_components)
            # lda.fit(data, state_seq)
        time_end = time.time()

    # reduction methods
    elif method in ['pca', 'umap']:
        time_start = time.time()
        for fname in fname_list:
            # data = np.load(os.path.join(raw_data_path, fname), allow_pickle=True)
            data, _ = load_data(f'data/{dataset}/raw/{fname}')
            if method == 'pca':
                data_reduced = PCA(n_components=n_components).fit_transform(data)
            elif method == 'umap':
                data_reduced = umap.UMAP(n_components=n_components).fit_transform(data)
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