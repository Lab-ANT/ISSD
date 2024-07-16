from issd import issd, ISSD
from miniutils import *
import os

# EXAMPLE OF ISSD, select on one time series
example_data, state_seq = load_data('data/MoCap/raw/86_02.npy')
selector = ISSD()
selector.compute_matrices([example_data], [state_seq])
selected_channels_qf = selector.get_qf_solution(3)
selected_channels_cf = selector.get_cf_solution(3)
true_cf_solution = selector.exhaustively_search_cf(3)
print(f'ISSD-QF selected channels: {selected_channels_qf}')
print(f'ISSD-CF selected channels: {selected_channels_cf}')
print(f'ISSD-CF selected channels: {true_cf_solution}')

# Select on multiple time series
fname_list = os.listdir('data/MoCap/raw')
fname_list.sort()
fname_list = fname_list[:len(fname_list)//2]
datalist = []
state_seq_list = []
for fname in fname_list:
    data, state_seq = load_data(f'data/MoCap/raw/{fname}')
    datalist.append(data)
    state_seq_list.append(state_seq)
selector = ISSD()
selector.compute_matrices(datalist, state_seq_list)
selected_channels_qf = selector.get_qf_solution(4)
selected_channels_cf = selector.get_cf_solution(4)
true_cf_solution = selector.exhaustively_search_cf(4)
print(f'ISSD-QF selected channels (milti-ts version): {selected_channels_qf}')
print(f'ISSD-CF selected channels (milti-ts version): {selected_channels_cf}')
print(f'ISSD-CF selected channels (milti-ts version): {true_cf_solution}')

# EXAMPLE OF ISSD, select on one time series
# example_data, state_seq = load_data('data/MoCap/raw/86_02.npy')
# selected_channels_qf, selected_channels_cf = issd([example_data], [state_seq], 4)
# print(f'ISSD-QF selected channels: {selected_channels_qf}')
# print(f'ISSD-CF selected channels: {selected_channels_cf}')

# Select on multiple time series
# fname_list = os.listdir('data/MoCap/raw')
# fname_list.sort()
# fname_list = fname_list[:len(fname_list)//2]
# datalist = []
# state_seq_list = []
# for fname in fname_list:
#     data, state_seq = load_data(f'data/MoCap/raw/{fname}')
#     datalist.append(data)
#     state_seq_list.append(state_seq)
# selected_channels_qf, selected_channels_cf = issd(datalist, state_seq_list, 4)
# print(f'ISSD-QF selected channels (milti-ts version): {selected_channels_qf}')
# print(f'ISSD-CF selected channels (milti-ts version): {selected_channels_cf}')
