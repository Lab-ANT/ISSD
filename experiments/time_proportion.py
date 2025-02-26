import sys
sys.path.append('.')
from issd import ISSD
from miniutils import *
import time
import os
import json

data_json = {
    'QF Searching': [],
    'CF Searching': [],
    'nntest': [],
    'QF & CF \nEvaluation': [],
}

for dataset in ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']:
    fname_list = os.listdir(f'data/{dataset}/raw')
    datalist = []
    state_seq_list = []
    for fname in fname_list:
        data, state_seq = load_data(f'data/{dataset}/raw/{fname}')
        datalist.append(data)
        state_seq_list.append(state_seq)
    
    selector = ISSD(n_jobs=20)
    start_nn = time.time()
    for i in range(5):
        selector.compute_matrices(datalist, state_seq_list)
    end_nn = time.time()
    start_qf = time.time()
    for i in range(5):
        selected_channels_qf = selector.get_qf_solution(4)
    end_qf = time.time()
    start_cf = time.time()
    for i in range(5):
        selected_channels_cf = selector.get_cf_solution(4)
    end_cf = time.time()
    start_inte = time.time()
    for i in range(5):
        selected_channels_cf = selector.inte_solution()
    end_inte = time.time()
    print(f'time taken for {dataset}: {end_inte-start_nn} seconds')
    data_json['QF Searching'].append((end_qf-start_qf)/5)
    data_json['CF Searching'].append((end_cf-start_cf)/5)
    data_json['nntest'].append((end_nn-start_nn)/5)
    data_json['QF & CF \nEvaluation'].append((end_inte-start_inte)/5)

with open('archive/other_output/time_consumed.json', 'w') as f:
    json.dump(data_json, f)
