"""
This script uses selection/reduction methods to select/reduce indicators on all datasets.
"""

import sys
sys.path.append('.')

import os
import tqdm
import argparse
import numpy as np
import pandas as pd
from miniutils import *
import time
# ISSD
from issd import issd
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
data_output_path = f'data/{dataset}/{method}' # path to save the selected data
os.makedirs(data_output_path, exist_ok=True)

if method not in ['issd', 'mi', 'ecs', 'ecp', 'lda', 'sfm', 'pca', 'umap']:
    raise ValueError(f'Unsupported method: {method}')

fname_list = os.listdir(raw_data_path)
fname_list.sort()
selected_channels = []

def mutual_info_selector(data, state_seq, n_components):
    score_list = []
    num_channels = data.shape[1]
    for i in range(num_channels):
        score=mutual_info_regression(data[:,i].reshape(-1,1), state_seq)[0]
        score_list.append(score)
    idx_sorted = np.argsort(score_list)[::-1]
    return list(idx_sorted[:n_components])

if method == 'issd':
    # precompute statistics for all time series
    for fname in tqdm.tqdm(fname_list):
        data, label = load_data(os.path.join(raw_data_path, fname))
        num_channels = data.shape[1]
        result = issd(data,
                    label,
                    n_components,
                    strategy='cf', # cf strategy
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

        print(f'qf: {selected_channels_qf}')
        print(f'cf: {selected_channels_cf}')

        score_qf = 0
        score_cf = 0
        for fname in fname_list_train:
            data, state_seq = load_data(f'data/{dataset}/raw/{fname}')
            reduced_data_issd_qf = data[:,selected_channels_qf]
            lda_issd_qf = LinearDiscriminantAnalysis(n_components=1).fit_transform(reduced_data_issd_qf, state_seq)
            score_qf += np.sum(mutual_info_regression(lda_issd_qf, state_seq))
            # lda_issd_qf = PCA(n_components=1).fit_transform(reduced_data_issd_qf, state_seq)
            # score_qf += np.sum(mutual_info_regression(lda_issd_qf, state_seq))
            # score_qf += np.sum(mutual_info_regression(reduced_data_issd_qf, state_seq))

            data, state_seq = load_data(f'data/{dataset}/raw/{fname}')
            reduced_data_issd_cf = data[:,selected_channels_cf]
            lda_issd_cf = LinearDiscriminantAnalysis(n_components=1).fit_transform(reduced_data_issd_cf, state_seq)
            score_cf += np.sum(mutual_info_regression(lda_issd_cf, state_seq))
            # lda_issd_cf = PCA(n_components=1).fit_transform(reduced_data_issd_cf, state_seq)
            # score_cf += np.sum(mutual_info_regression(lda_issd_cf, state_seq))
            # score_cf += np.sum(mutual_info_regression(reduced_data_issd_cf, state_seq))

        s_qf = score_qf
        s_cf = score_cf
        if s_qf >= s_cf:
            selected_channels = selected_channels_qf
        else:
            selected_channels = selected_channels_cf
        print(f'selected channels: {selected_channels}')
        
        os.makedirs(data_output_path+'-qf', exist_ok=True)
        os.makedirs(data_output_path+'-cf', exist_ok=True)
        for fname in fname_list_test:
            print('issd', s_qf, s_cf, fname)
            data = np.load(f'data/{dataset}/raw/{fname}', allow_pickle=True)
            np.save(f'data/{dataset}/issd/{fname}', data[:,selected_channels+[-1]])
            np.save(f'data/{dataset}/issd-qf/{fname}', data[:,selected_channels_qf+[-1]])
            np.save(f'data/{dataset}/issd-cf/{fname}', data[:,selected_channels_cf+[-1]])

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
            print(result[:n_components])
            for fn_test in fname_list_test:
                data, state_seq = load_data(os.path.join(raw_data_path, fn_test))
                data_reduced = data[:,result]
                data_reduced = np.vstack((data_reduced.T, state_seq)).T
                np.save(os.path.join(f'data/{dataset}/{method}', fn_test), data_reduced)
        # MI
        elif method == 'mi':
            result = mutual_info_selector(data, state_seq, n_components)
            print(result)
            for fn_test in fname_list_test:
                data, state_seq = load_data(os.path.join(raw_data_path, fn_test))
                data_reduced = data[:,result]
                data_reduced = np.vstack((data_reduced.T, state_seq)).T
                np.save(os.path.join(f'data/{dataset}/{method}', fn_test), data_reduced)
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
            print(result)
            for fn_test in fname_list_test:
                data, state_seq = load_data(os.path.join(raw_data_path, fn_test))
                data_reduced = data[:,result]
                data_reduced = np.vstack((data_reduced.T, state_seq)).T
                np.save(os.path.join(f'data/{dataset}/{method}', fn_test), data_reduced)
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
                np.save(os.path.join(f'data/{dataset}/{method}', fn_test), data_reduced)
    time_end = time.time()
    print(f'time taken for {method}: {time_end-time_start} seconds')

# reduction methods
elif method in ['pca', 'umap']:
    time_start = time.time()
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
    time_end = time.time()
    print(f'time taken for {method}: {time_end-time_start} seconds')