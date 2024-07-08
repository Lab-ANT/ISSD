"""
Test downstream methods on all datasets with different selection/reduction methods.
"""

import sys
sys.path.append('.')
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import tqdm
from miniutils import load_data
from sklearn.metrics import normalized_mutual_info_score
# suppress the future warnings of sklearn, which is harmless.
import warnings
warnings.filterwarnings("ignore")
import argparse

from downstream_methods.downstream_adaper import DownstreamMethodAdaper
# receive params
parser = argparse.ArgumentParser()
parser.add_argument('downstream', type=str, help='The downstream method name')
parser.add_argument('dataset', type=str, help='The dataset name')
args = parser.parse_args()
dmethod = args.downstream
dataset = args.dataset

script_path = os.path.dirname(__file__)

if dmethod not in ['time2state', 'ticc', 'e2usd']:
    print('The downstream method does not exist. Please check the downstream method name.')
    exit()

f_list = os.listdir(os.path.join(script_path, f'../data/{dataset}/raw/'))
f_list = list(f_list)

overall_score_list = []
for fname in f_list:
    data, label = load_data(f'output/sfm/{dmethod}/{dataset}/{fname}')

    num_channels = data.shape[1]-1
    channel_score_list = []
    for channel_id in tqdm.trange(num_channels):
        channel_prediction = data[:, channel_id]
        nmi = normalized_mutual_info_score(channel_prediction, label)
        channel_score_list.append(nmi)

    # idx_sorted = np.argsort(channel_score_list)[::-1]
    # print(f'{fname}: {idx_sorted}')
    channel_score_list = np.array(channel_score_list)
    overall_score_list.append(channel_score_list)
overall_score_list = np.array(overall_score_list).T

print(overall_score_list.shape)
overall_score_list = np.mean(overall_score_list, axis=1)
print(overall_score_list.shape)

sorted_idx = np.argsort(overall_score_list)[::-1]

selected_channels = sorted_idx[:4]

os.makedirs(f'data/MoCap/t2s/', exist_ok=True)
for fname in f_list:
    data, label = load_data(f'data/{dataset}/raw/{fname}')
    reduced_data = data[:, selected_channels]
    data = np.vstack([reduced_data.T, label]).T
    print(data.shape)
    np.save(f'data/{dataset}/t2s/{fname}', data)

