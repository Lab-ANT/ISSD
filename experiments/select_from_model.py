"""
Created by Chengyu on 2023/11/15.
Test downstream methods on all datasets with different selection/reduction methods.
"""

import sys
sys.path.append('.')
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import tqdm
from miniutils import load_data
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
output_path = os.path.join(script_path, f'../output/sfm/{dmethod}/{dataset}/')
os.makedirs(output_path, exist_ok=True)

for fname in f_list:
    data, label = load_data(os.path.join(script_path, f'../data/{dataset}/raw/', fname))
    data = StandardScaler().fit_transform(data)

    num_channels = data.shape[1]
    channel_prediction_list = [] # save the prediction results of each channel
    for channel_id in tqdm.trange(num_channels):
        channel_data = data[:, channel_id].reshape(-1,1)
        estimator = DownstreamMethodAdaper(dmethod)
        # Note these methods are unsupervised
        # the label will not be used in the downstream methods.
        # we just use it to calculate #states
        prediction = estimator.fit_transform(data, label)

        # when the prediction is None, it means the method fails to run
        if prediction is None:
            print(f'Error in {fname}')
            # generate a pseudo prediction of the same length as the label
            # use value 0
            prediction = np.zeros_like(label)

        min_len = min(len(prediction), len(label))
        prediction = prediction[:min_len]
        label = label[:min_len]
        channel_prediction_list.append(prediction)
    # save the prediction results and ground truth
    channel_prediction_list.append(label)
    result = np.vstack(channel_prediction_list).T # the last channel is label
    print(result.shape)
    np.save(os.path.join(output_path, fname), result)