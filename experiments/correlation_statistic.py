"""
Statistical analysis of the correlation distribution
between the channels in each dataset.
"""

import os
import sys
sys.path.append('.')
from miniutils import *

path = 'data/'
dataset_list = ['MoCap', 'USC-HAD', 'PAMAP2', 'SynSeg', 'ActRecTut']

for d in dataset_list:
    fname_list = os.listdir(f'data/{d}/raw')
    for fname in fname_list:
        data = np.load(f'data/{d}/raw/{fname}', allow_pickle=True)
        # calculate the correlation between the channels
        data = data[:,:-1] # exclude the label
        corr = np.corrcoef(data.T)
        # pad nan with 0, two constant channels can yield nan
        idx_nan = np.isnan(corr)
        corr[idx_nan] = 0
        # only consider the upper triangle, excluding the diagonal
        corr = corr[np.triu_indices(corr.shape[0], k=1)]
        minimum = np.min(corr)
        maximum = np.max(corr)
        print(d, minimum, maximum)