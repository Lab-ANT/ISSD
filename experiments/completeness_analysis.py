import sys
sys.path.append('.')
import os
import numpy as np
from miniutils import *
from issd import ISSD

# datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']
datasets = ['MoCap']

os.makedirs('completeness_analysis', exist_ok=True)

for d in datasets:
    # Select on multiple time series
    fname_list = os.listdir(f'data/{d}/raw')
    fname_list.sort()
    # fname_list = fname_list[len(fname_list)//2:]
    # fname_list = fname_list[:len(fname_list)//2]
    datalist = []
    state_seq_list = []
    for fname in fname_list:
        data, state_seq = load_data(f'data/{d}/raw/{fname}')
        datalist.append(data)
        state_seq_list.append(state_seq)
    selector = ISSD()
    selector.compute_matrices(datalist, state_seq_list)
    selector.compute_completeness_quality()
    selector.get_cf_solution(4)
    cf_solution = selector.cf_solution
    print(cf_solution)
    matrices = selector.matrices
    true_matrices = selector.true_matrices
    ts_ch = []
    theoritical_highest_list = []
    for ts_idx in range(len(matrices)):
        ch_indicator = []
        for ch_idx in cf_solution:
            matrix = matrices[ts_idx][ch_idx]
            true_matrix = true_matrices[ts_idx]
            idx_inner = np.argwhere(true_matrix==True)
            idx_inter = np.argwhere(true_matrix==False)
            max_inner = np.max(matrix[idx_inner])
            min_inter = np.min(matrix[idx_inter])
            indicator_matrix = matrix > max_inner
            ch_indicator.append(indicator_matrix)
        theoritical_highest_list.append(len(idx_inter))
        ch_indicator = np.stack(ch_indicator, axis=0)
        # print(ch_indicator.shape)
        ts_ch.append(ch_indicator)
    print(theoritical_highest_list)
    ts_completeness = []
    for ts_idx, ch_indicator in enumerate(ts_ch):
        iter_list = []
        # print(ch_indicator.shape)
        for i in range(len(cf_solution)):
            c = np.sum(matrix_OR(ch_indicator[:i+1]))/theoritical_highest_list[ts_idx]
            iter_list.append(c)
        ts_completeness.append(iter_list)
    ts_completeness = np.array(ts_completeness)
    # print(ts_completeness.shape)
    np.save(f'completeness_analysis/{d}_iter.npy', ts_completeness)
    np.save(f'completeness_analysis/{d}_completeness.npy', selector.completeness)
    np.save(f'completeness_analysis/{d}_quality.npy', selector.quality)
    print(f'Completed {d}')