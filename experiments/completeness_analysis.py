import sys
sys.path.append('.')
import os
import numpy as np
from miniutils import load_data
from issd import ISSD

datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']

os.makedirs('completeness_analysis', exist_ok=True)

for d in datasets:
    # Select on multiple time series
    fname_list = os.listdir(f'data/{d}/raw')
    fname_list.sort()
    datalist = []
    state_seq_list = []
    for fname in fname_list:
        data, state_seq = load_data(f'data/{d}/raw/{fname}')
        datalist.append(data)
        state_seq_list.append(state_seq)
    selector = ISSD()
    selector.compute_matrices(datalist, state_seq_list)
    selector.get_completeness_quality()
    # print(selector.completeness.shape)
    # print(selector.quality.shape)
    np.save(f'completeness_analysis/{d}_completeness.npy', selector.completeness)
    np.save(f'completeness_analysis/{d}_quality.npy', selector.quality)
    print(f'Completed {d}')