from issd import ISSD
from miniutils import *
import os

# EXAMPLE OF ISSD, select on one time series
example_data, state_seq = load_data('data/MoCap/raw/86_02.npy')
selector = ISSD(n_jobs=5)
selector.fit([example_data], [state_seq], 4)
print(f'ISSD-QF selected channels: {selector.qf_solution}')
print(f'ISSD-CF selected channels: {selector.cf_solution}')
print(f'ISSD-Inte selected channels: {selector.solution}')

# You can also only use cf or qf strategy
selector = ISSD(n_jobs=5)
selector.compute_matrices([example_data], [state_seq])
selected_channels_qf = selector.get_qf_solution(4)
selected_channels_cf = selector.get_cf_solution(4)
selected_channels_inte = selector.inte_solution()
print(f'ISSD-QF selected channels: {selected_channels_qf}')
print(f'ISSD-CF selected channels: {selected_channels_cf}')
print(f'ISSD-Inte selected channels: {selected_channels_inte}')

# EXAMPLE OF ISSD, select on multiple time series
fname_list = os.listdir('data/MoCap/raw')
fname_list.sort()
fname_list = fname_list[:len(fname_list)//2]
datalist = []
state_seq_list = []
for fname in fname_list:
    data, state_seq = load_data(f'data/MoCap/raw/{fname}')
    datalist.append(data)
    state_seq_list.append(state_seq)
selector = ISSD(n_jobs=5)
selector.fit(datalist, state_seq_list, 4)
print(f'ISSD-QF selected channels (milti-ts version): {selector.qf_solution}')
print(f'ISSD-CF selected channels (milti-ts version): {selector.cf_solution}')
print(f'ISSD-Inte selected channels (milti-ts version): {selector.solution}')