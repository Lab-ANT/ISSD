import sys
sys.path.append('.')
import numpy as np
import os
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from miniutils import purity_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('metric', type=str, default='nmi')
args = parser.parse_args()
metric = args.metric

datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'SynSeg', 'USC-HAD']
dmthods = ['time2state', 'e2usd', 'ticc', 'autoplait']

result_json = {}
for m in ['time2state', 'e2usd']:
    for d in datasets:
        for corr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # for corr in [0.2, 0.3]:
            nmi_list = []
            for i in range(1, 6):
                path = f'archive/corr_effect/{d}/{corr}/{m}{i}/issd/'
                fname_list = os.listdir(path)
                nmi_list_on_dataset = []
                for fname in fname_list:
                    result = np.load(path + fname)
                    if metric == 'nmi':
                        nmi = normalized_mutual_info_score(result[0, :], result[1, :])
                    elif metric == 'ari':
                        nmi = adjusted_rand_score(result[0, :], result[1, :])
                    elif metric == 'purity':
                        nmi = purity_score(result[0, :], result[1, :])
                    # print(nmi)
                    nmi_list_on_dataset.append(nmi)
                nmi_on_dataset = np.mean(nmi_list_on_dataset)
                nmi_list.append(nmi_on_dataset)
            mean_nmi = np.mean(nmi_list)
            print(m, d, corr, mean_nmi)

for m in ['ticc', 'autoplait']:
    for d in datasets:
        for corr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # for corr in [0.2, 0.3]:
            path = f'archive/corr_effect/{d}/{corr}/{m}/issd/'
            fname_list = os.listdir(path)
            nmi_list_on_dataset = []
            for fname in fname_list:
                result = np.load(path + fname)
                # nmi = normalized_mutual_info_score(result[0, :], result[1, :])
                if metric == 'nmi':
                    nmi = normalized_mutual_info_score(result[0, :], result[1, :])
                elif metric == 'ari':
                    nmi = adjusted_rand_score(result[0, :], result[1, :])
                elif metric == 'purity':
                    nmi = purity_score(result[0, :], result[1, :])
                nmi_list_on_dataset.append(nmi)
            nmi_on_dataset = np.mean(nmi_list_on_dataset)
            print(m, d, corr, nmi_on_dataset)