"""
Test downstream methods on all datasets with different selection/reduction methods.
"""

import sys
sys.path.append('.')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import os
import tqdm
# suppress the future warnings of sklearn, which is harmless.
import warnings
warnings.filterwarnings("ignore")
import argparse

from downstream_methods.downstream_adaper import DownstreamMethodAdaper
# receive params
parser = argparse.ArgumentParser()
parser.add_argument('downstream', type=str, help='The downstream method name')
parser.add_argument('dataset', type=str, help='The dataset name')
parser.add_argument('method', type=str, help='The method name')
args = parser.parse_args()
dmethod = args.downstream
dataset = args.dataset
method = args.method

script_path = os.path.dirname(__file__)

print(f'Processing {dataset} using {method} with {dmethod}')

# if not os.path.exists(os.path.join(script_path, f'../data/{dataset}/{method}/')):
#     print('The data does not exist. Please check the dataset and method name.')
#     exit()

# if dmethod not in ['time2state', 'ticc', 'e2usd']:
    # print('The downstream method does not exist. Please check the downstream method name.')
    # exit()

f_list = os.listdir(os.path.join(script_path, f'../data/{dataset}/raw/'))
f_list = list(f_list)
output_path = os.path.join(script_path, f'../output/results/{dmethod}/{dataset}/{method}/')
os.makedirs(output_path, exist_ok=True)
nmi_list = []
for fname in tqdm.tqdm(f_list):
    data = np.load(os.path.join(script_path, f'../data/{dataset}/{method}/', fname))
    label = data[:,-1]
    data = data[:,:-1]
    data = StandardScaler().fit_transform(data)
    num_states = len(set(label))

    estimator = DownstreamMethodAdaper(dmethod)
    # Note these methods are unsupervised
    # the label will not be used in the downstream methods.
    # we just use it to calculate #states
    prediction = estimator.fit_transform(data, label)

    if prediction is None:
        print(f'Error in {fname}')
        continue

    min_len = min(len(prediction), len(label))
    prediction = prediction[:min_len]
    label = label[:min_len]

    nmi = normalized_mutual_info_score(label, prediction)
    nmi_list.append(nmi)
    # save the prediction results and ground truth
    np.save(os.path.join(output_path, fname), np.vstack((prediction, label)))
print(f'NMI of {dmethod} on {dataset} with {method}: {np.mean(nmi_list)}')