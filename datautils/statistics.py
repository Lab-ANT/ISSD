"""
Created by Chengyu.
statistics of the dataset.
"""

import sys
sys.path.append('.')
from miniutils import *
import numpy as np
import os
import prettytable as pt

datasets = ['PAMAP2', 'SynSeg', 'MoCap', 'ActRecTut', 'USC-HAD']

pretty_table = pt.PrettyTable()
columns = ['length (k)', '# channels', '# states', '# segments', '# duration (k)', 'total states']
pretty_table.field_names = ['Datasets'] + columns

for dname in datasets:
    dataset_path = f'data/{dname}/raw/'
    if not os.path.exists(dataset_path):
        print(f'{dname} does not exist.')
        continue
    f_list = os.listdir(dataset_path)
    length_list = []
    num_state_list = []
    num_segs_list = []
    seg_len_list = []
    set_states = []
    for fname in f_list:
        # data = np.load(os.path.join(dataset_path, fname), allow_pickle=True)
        data, state_seq = load_data(os.path.join(dataset_path, fname))
        non_trival_idx = exclude_trival_segments(state_seq, 50)
        data = data[non_trival_idx]
        state_seq = state_seq[non_trival_idx]
        num_channels = data.shape[1]
        length = data.shape[0]
        num_state = len(np.unique(state_seq))
        cps = find_cut_points_from_state_seq(state_seq)
        num_segs = len(cps)
        seg_len_list+=np.diff(cps).tolist()
        length_list.append(length)
        num_state_list.append(num_state)
        num_segs_list.append(num_segs)
        set_states += list(np.unique(state_seq))
    min_num_states = min(num_state_list)
    max_num_states = max(num_state_list)
    min_num_segs = min(num_segs_list)
    max_num_segs = max(num_segs_list)
    min_length = min(length_list)/1000
    max_length = max(length_list)/1000
    min_seg_len = min(seg_len_list)/1000
    max_seg_len = max(seg_len_list)/1000
    total_states = len(set(set_states))
    pretty_table.add_row([dname,
                          f'{min_length}~{max_length}',
                          num_channels,
                          f'{min_num_states}~{max_num_states}',
                          f'{min_num_segs}~{max_num_segs}',
                          f'{min_seg_len}~{max_seg_len}',
                          total_states])
print(pretty_table)