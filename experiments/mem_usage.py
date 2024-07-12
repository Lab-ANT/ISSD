import sys
sys.path.append('.')
from memory_profiler import memory_usage
import time

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
    
    def my_function():
        selector = ISSD()
        selector.compute_matrices(datalist, state_seq_list)
        selected_channels_qf = selector.get_qf_solution(4)
        selected_channels_cf = selector.get_cf_solution(4)
    mem_usage = memory_usage((my_function, ))
    print(f"Memory usage: {max(mem_usage) - min(mem_usage)} MB")

# with open('time_consumed.json', 'w') as f:
#     json.dump(data_json, f)
