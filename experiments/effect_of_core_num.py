import sys
sys.path.append('.')
from issd import ISSD
from miniutils import *
import time
import os
import json

# EXAMPLE OF ISSD, select on one time series
example_data, state_seq = load_data('data/MoCap/raw/86_02.npy')
# print(example_data.shape, state_seq.shape)
example_data = np.concatenate([example_data, example_data, example_data, example_data], axis=1)
example_data = example_data[:,:100]
# print(example_data.shape)

time_consumption = []
for i in range(5):
    time_list = []
    for num_cores in range(1, 21):
        selector = ISSD(n_jobs=num_cores)
        start = time.time()
        selector.compute_matrices([example_data], [state_seq])
        selected_channels_qf = selector.get_qf_solution(4)
        selected_channels_cf = selector.get_cf_solution(4)
        selector.inte_solution()
        end = time.time()
        print(f'computation time: {end-start} seconds')
        time_list.append(end-start)
    time_consumption.append(time_list)
time_consumption = np.mean(time_consumption, axis=0)
np.save('effect_of_core_num.npy', time_consumption)
