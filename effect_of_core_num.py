from issd import ISSD
from miniutils import *
import time
import os
import json

# EXAMPLE OF ISSD, select on one time series
example_data, state_seq = load_data('data/MoCap/raw/86_02.npy')

time_consumption = []
for i in range(2):
    time_list = []
    for num_cores in range(1, 11):
        selector = ISSD(n_jobs=num_cores)
        start = time.time()
        selector.compute_matrices([example_data], [state_seq])
        selected_channels_qf = selector.get_qf_solution(4)
        selected_channels_cf = selector.get_cf_solution(4)
        end = time.time()
        print(f'computation time: {end-start} seconds')
        time_list.append(end-start)
    time_consumption.append(time_list)
time_consumption = np.mean(time_consumption, axis=0)
np.save('effect_of_core_num.npy', time_consumption)
