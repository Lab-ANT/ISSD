from issd import ISSD
from miniutils import *
import time
import os
import json

data_json = {
    'QF': [],
    'CF': [],
    'NN': []
}

for dataset in ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']:
    fname_list = os.listdir(f'data/{dataset}/raw')
    datalist = []
    state_seq_list = []
    for fname in fname_list:
        data, state_seq = load_data(f'data/{dataset}/raw/{fname}')
        datalist.append(data)
        state_seq_list.append(state_seq)
    
    selector = ISSD()
    start_nn = time.time()
    selector.compute_matrices(datalist, state_seq_list)
    end_nn = time.time()
    start_qf = time.time()
    selected_channels_qf = selector.get_qf_solution(4)
    end_qf = time.time()
    start_cf = time.time()
    selected_channels_cf = selector.get_cf_solution(4)
    end_cf = time.time()
    print(f'time taken for {dataset} iteration: {end_cf-start_nn} seconds')
    data_json['QF'].append(end_qf-start_qf)
    data_json['CF'].append(end_cf-start_cf)
    data_json['NN'].append(end_nn-start_nn)

with open('time_consumed.json', 'w') as f:
    json.dump(data_json, f)
