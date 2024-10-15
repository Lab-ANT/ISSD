import sys
sys.path.append('.')
import os
import numpy as np
from miniutils import load_data, find_cut_points_from_state_seq
from issd import ISSD
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

os.makedirs('completeness_analysis', exist_ok=True)

"""
Selection
"""
# # Select on multiple time series
# data, state_seq = load_data('data/MoCap/raw/86_02.npy')
# selector = ISSD()
# selector.compute_matrices([data], [state_seq])
# # selector.compute_completeness_quality()
# selector.get_cf_solution(4)
# print(selector.cf_solution)

"""
Plot
"""
data, state_seq = load_data('data/MoCap/raw/86_02.npy')
selected_channels = [58, 59, 48, 56]
cps = find_cut_points_from_state_seq(state_seq)

# plt.style.use('classic')
results = []
fig, ax = plt.subplots(nrows=4, figsize=(8, 4))
for i, idx in enumerate(selected_channels):
    channel = data[:, idx]
    # scale to 0-1
    channel = (channel - channel.min()) / (channel.max() - channel.min())
    ax[i].plot(channel)
    # ax[i].set_title(f'Channel {idx}')
    results.append(idx)
    ax[i].set_xticks(f'Result set: {results}')

    for cp in cps:
        ax[i].axvline(cp, color='r', linestyle='--')

plt.tight_layout()
plt.savefig('completeness_analysis/selected_channels.png')