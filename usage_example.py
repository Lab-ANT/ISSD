"""
Created by Chengyu on 2024/3/10.
This is an example of how to use the ISSD and IRSD.
"""

from issd import issd
from miniutils import *
import os

# EXAMPLE OF ISSD, select on one time series
example_data, state_seq = load_data('data/MoCap/raw/86_02.npy')
selected_channels_qf, selected_channels_cf = issd([example_data], [state_seq], 4)
print(f'ISSD-QF selected channels: {selected_channels_qf}')
print(f'ISSD-CF selected channels: {selected_channels_cf}')

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
selected_channels_qf, selected_channels_cf = issd(datalist, state_seq_list, 4)
print(f'ISSD-QF selected channels: {selected_channels_qf}')
print(f'ISSD-CF selected channels: {selected_channels_cf}')
