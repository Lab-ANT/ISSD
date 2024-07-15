from issd import ISSD
from miniutils import *
import time
import os
import json
import timeit
import numpy as np

# EXAMPLE OF ISSD, select on one time series
example_data, state_seq = load_data('data/MoCap/raw/86_02.npy')
example_data2, state_seq2 = load_data('data/MoCap/raw/86_02.npy')
example_data = np.concatenate([example_data, example_data2], axis=1)
# state_seq = np.concatenate([state_seq, state_seq2], axis=0)

print(example_data.shape, state_seq.shape)
time_consumption = []
for i in range(2):
    time_list = []
    for num_cores in range(1, 31):
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

# time_consumption = []
# for num_cores in range(1, 31):
#     def func():
#         selector = ISSD(n_jobs=num_cores)
#         selector.compute_matrices([example_data], [state_seq])
#         selected_channels_qf = selector.get_qf_solution(4)
#         selected_channels_cf = selector.get_cf_solution(4)
#     t = timeit.Timer(func)
#     time_used = t.timeit(number=5)
#     time_consumption.append(time_used)
#     print(f'computation time: {time_used} seconds')
# time_consumption = np.array(time_consumption)
# np.save('effect_of_core_num2.npy', time_consumption)