import sys
sys.path.append('.')
import os
import numpy as np
from miniutils import load_data, find_cut_points_from_state_seq, reorder_label
from issd import ISSD
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

os.makedirs('completeness_analysis', exist_ok=True)

"""
Selection
"""
fname = '86_08.npy'

# Select on multiple time series
data, state_seq = load_data(f'data/MoCap/raw/{fname}')
selector = ISSD()
selector.compute_matrices([data], [state_seq])
# selector.compute_completeness_quality()
selector.get_cf_solution(4)
print(selector.cf_solution)
selected_channels = selector.cf_solution

"""
Plot
"""
data, state_seq = load_data(f'data/MoCap/raw/{fname}')
state_seq = state_seq.reshape(-1, 1)
# selected_channels = [58, 59, 48, 56]
state_seq = reorder_label(state_seq)

cps = find_cut_points_from_state_seq(state_seq)
length = data.shape[0]

# plt.style.use('classic')
results = []
fig, ax = plt.subplots(nrows=4, figsize=(8, 4))
for i, idx in enumerate(selected_channels):
    channel = data[:, idx]
    # scale to 0-1
    channel = (channel - channel.min()) / (channel.max() - channel.min())
    ax[i].plot(channel)
    results.append(idx)
    # ax[i].set_xticks(f'Result set: {results}')

    for cp in cps:
        ax[i].axvline(cp, color='black', linestyle='--')
    ax[i].set_xlim(0, length)
    # remove x ticks
    if i < len(selected_channels)-1:
        ax[i].set_xticks([])

    # imshow state sequence at the same ax. y range is 0-1
    ax[i].imshow(state_seq.T, aspect='auto', cmap='tab10', alpha=0.5, extent=[0, length, 0, 1])

plt.tight_layout()
plt.savefig('completeness_analysis/selected_channels.png')