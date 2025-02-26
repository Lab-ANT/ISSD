"""
Created by Chengyu.
Used for testing the effect of core number on the computation time.
The core number is set from 1 to 20.
If your computer has no more than 20 cores,
you can modify the MAX_NUM_CORES to the number of cores you have.
The computation time is the average of 5 runs,
you can modify the NUM_RUNS to the number of runs you want.
The result is saved in 'archive/other_output/effect_of_core_num.npy'.
"""

import sys
sys.path.append('.')
from issd import ISSD
from miniutils import *
import time
import os

MAX_NUM_CORES = 10
NUM_RUNS = 5

# Create the directory for saving the results
os.makedirs('archive/other_output', exist_ok=True)

# EXAMPLE OF ISSD, select on one time series
example_data, state_seq = load_data('data/MoCap/raw/86_02.npy')
# print(example_data.shape, state_seq.shape)
example_data = np.concatenate([example_data, example_data, example_data, example_data], axis=1)
example_data = example_data[:,:100]
# print(example_data.shape)

time_consumption = []
for i in range(NUM_RUNS):
    time_list = []
    for num_cores in range(1, MAX_NUM_CORES+1):
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
np.save('archive/other_output/effect_of_core_num.npy', time_consumption)
