"""
Created by Chengyu.
The channel sensitivity experiment.
"""

import sys
sys.path.append('.')
from downstream_methods.downstream_adaper import DownstreamMethodAdaper
from sklearn.metrics import normalized_mutual_info_score
from miniutils import load_data
import os
import numpy as np
# suppress the future warnings of sklearn, which is harmless.
import warnings
warnings.filterwarnings("ignore")
import argparse

# Separately run time2state and e2usd!!!!!!
# or they will conflict with each other!!!!!
# this is because downstream adaper does not solve their conflicts.
# continuous run them will import the same module name, which will cause conflicts.

argparser = argparse.ArgumentParser()
# positional arguments
argparser.add_argument('method', type=str, help='method to run')
method = argparser.parse_args().method

# method = 'e2usd' # time2state, e2usd, ticc

for i in range(1,11):
    # case study data is only one for each ratio.
    # run them 5 times to get the average performance.
    # autoplait and ticc does not need to run 5 times.
    # because they are deterministic.
    os.makedirs(f'output/sensitivity/{method}/', exist_ok=True)
    if method in ['time2state', 'e2usd']:
        for j in range(1,6):
            data, state_seq = load_data(f'data/CaseStudy/case{i}.npy')
            print(i, data.shape)
            estimator = DownstreamMethodAdaper(method)
            # Note these methods are unsupervised
            # the state_seq will not be used in the downstream methods.
            # we just use it to calculate #states
            prediction = estimator.fit_transform(data, state_seq)
            min_len = min(len(prediction), len(state_seq))
            prediction = prediction[:min_len]
            state_seq = state_seq[:min_len]
            nmi = normalized_mutual_info_score(state_seq, prediction)
            print(method, nmi)
            np.save(f'output/sensitivity/{method}/case{i}-{j}.npy', np.vstack((prediction, state_seq)))
    elif method == 'ticc':
        data, state_seq = load_data(f'data/CaseStudy/case{i}.npy')
        print(i, data.shape)
        estimator = DownstreamMethodAdaper(method)
        # Note these methods are unsupervised
        # the state_seq will not be used in the downstream methods.
        # we just use it to calculate #states
        prediction = estimator.fit_transform(data, state_seq)
        min_len = min(len(prediction), len(state_seq))
        prediction = prediction[:min_len]
        state_seq = state_seq[:min_len]
        nmi = normalized_mutual_info_score(state_seq, prediction)
        print(method, nmi)
        np.save(f'output/sensitivity/{method}/case{i}.npy', np.vstack((prediction, state_seq)))

# VISUALIZE THE CASE STUDY DATA
# results are saved in output/plots
# import matplotlib.pyplot as plt
# os.makedirs('output/plots/', exist_ok=True)
# for i in range(1,11):
#     fig, ax = plt.subplots(nrows=10, figsize=(10,10))
#     data, state_seq = load_data(f'data/CaseStudy/case{i}.npy')
#     for j in range(1,11):
#         ax[j-1].plot(data[:,j-1])
#     plt.savefig(f'output/plots/case{i}.png')