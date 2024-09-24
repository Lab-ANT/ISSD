import sys
sys.path.append('.')
import os
from miniutils import load_data
from issd import ISSD

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
selector = ISSD(num_samples=2)
selector.compute_matrices(datalist, state_seq_list)
selector.get_completeness()
# selected_channels_qf = selector.get_qf_solution(4)
# selected_channels_cf = selector.get_cf_solution(4)
# true_cf_solution = selector.exhaustively_search_cf(4)
# print(f'ISSD-QF selected channels (milti-ts version): {selected_channels_qf}')
# print(f'ISSD-CF selected channels (milti-ts version): {selected_channels_cf}')
# print(f'ISSD-CF selected channels (milti-ts version): {true_cf_solution}')