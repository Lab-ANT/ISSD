# import sys
# sys.path.append('.')
# import os
# import numpy as np
# from miniutils import load_data, find_cut_points_from_state_seq, reorder_label
# from issd import ISSD
# import matplotlib.pyplot as plt
# import sys
# sys.path.append('.')

# os.makedirs('completeness_analysis', exist_ok=True)

# """
# Selection
# """
# fname = 'data/MoCap/raw/86_08.npy'
# # fname = 'data/SynSeg/raw/synthetic2.npy'

# # Select on multiple time series
# # data, state_seq = load_data(fname)
# # selector = ISSD()
# # selector.compute_matrices([data], [state_seq])
# # # selector.compute_completeness_quality()
# # selector.get_cf_solution(4)
# # print(selector.cf_solution)
# # selected_channels = selector.cf_solution

# """
# Plot
# """
# data, state_seq = load_data(fname)
# state_seq = state_seq.reshape(-1, 1)
# # selected_channels = [58, 59, 48, 56]
# selected_channels = [58, 28, 38, 55]
# state_seq = reorder_label(state_seq)

# cps = find_cut_points_from_state_seq(state_seq)
# length = data.shape[0]

# # plt.style.use('classic')
# results = []
# channel_set = []
# # fig, ax = plt.subplots(nrows=4, figsize=(8, 4))
# fig, ax = plt.subplots(nrows=4, ncols=2, gridspec_kw={'width_ratios': [8, 1]}, figsize=(8, 4))
# for i, idx in enumerate(selected_channels):
#     channel = data[:, idx]
#     # scale to 0-1
#     channel = (channel - channel.min()) / (channel.max() - channel.min())
#     channel_set.append(channel)

#     # imshow state sequence at the same ax. y range is 0-1
#     ax[i, 0].imshow(state_seq.T, aspect='auto', cmap='tab10', alpha=0.5, extent=[0, length, 0, 1])

#     # plot the former results in the same ax, gray
#     for k in range(i):
#         ax[i, 0].plot(channel_set[k], color='gray', alpha=0.4)
#     ax[i, 0].plot(channel)
#     results.append(idx)
#     # ax[i].set_xticks(f'Result set: {results}')

#     for cp in cps:
#         ax[i, 0].axvline(cp, color='black', linestyle='--')
#     ax[i, 0].set_xlim(0, length)
#     # remove x ticks
#     if i < len(selected_channels)-1:
#         ax[i, 0].set_xticks([])

#     # Generate a random matrix for testing and display it on the right
#     random_matrix = np.random.rand(10, 10)  # Example matrix with 10 columns
#     random_matrix = random_matrix>0.5
#     # ax[i, 1].imshow(random_matrix, aspect='auto', cmap='gray')
#     ax[i, 1].imshow(random_matrix, cmap='gray')
#     # ax[i, 1].axis('off')  # Turn off the axis for the matrix plot
#     ax[i, 1].set_xticks([])  # Turn off the x-ticks for the matrix plot
#     ax[i, 1].set_yticks([])  # Turn off the y-ticks for the matrix plot

# plt.tight_layout()
# plt.savefig('completeness_analysis/selected_channels.png')

import sys
sys.path.append('.')
import os
import numpy as np
from miniutils import load_data, find_cut_points_from_state_seq, reorder_label
from issd import ISSD
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
sys.path.append('.')

os.makedirs('completeness_analysis', exist_ok=True)

"""
Selection
"""
fname = 'data/MoCap/raw/86_08.npy'
# fname = 'data/SynSeg/raw/synthetic2.npy'

# Select on multiple time series
data, state_seq = load_data(fname)
selector = ISSD()
selector.compute_matrices([data], [state_seq])
# selector.compute_completeness_quality()
selector.get_cf_solution(4)
print(selector.cf_solution)
selected_channels = selector.cf_solution

"""
Plot
"""
data, state_seq = load_data(fname)
state_seq = state_seq.reshape(-1, 1)
# selected_channels = [58, 59, 48, 56]
# selected_channels = [58, 28, 38, 55]
state_seq = reorder_label(state_seq)

cps = find_cut_points_from_state_seq(state_seq)
length = data.shape[0]

# plt.style.use('classic')
results = []
channel_set = data[:, selected_channels]
channel_set = StandardScaler().fit_transform(channel_set)
# scale to 0-1
channel_set = (channel_set - channel_set.min()) / (channel_set.max() - channel_set.min())
channel_set = [channel_set[:, i] for i in range(channel_set.shape[1])]

# fig, ax = plt.subplots(nrows=4, figsize=(8, 4))
fig, ax = plt.subplots(nrows=4, ncols=2, gridspec_kw={'width_ratios': [8, 1]}, figsize=(8, 4))
for i, idx in enumerate(selected_channels):
    channel = channel_set[i]

    # plot the former results in the same ax, gray
    for k in range(i):
        # ax[i, 0].plot(channel_set[k], color='gray', alpha=0.6)
        ax[i, 0].plot(channel_set[k])
    ax[i, 0].plot(channel)
    results.append(idx)
    # ax[i].set_xticks(f'Result set: {results}')

    for cp in cps:
        ax[i, 0].axvline(cp, color='black', linestyle='--')
    ax[i, 0].set_xlim(0, length)
    # remove x ticks
    if i < len(selected_channels)-1:
        ax[i, 0].set_xticks([])

    # imshow state sequence at the same ax. y range is 0-1
    ax[i, 0].imshow(state_seq.T, aspect='auto', cmap='tab10', alpha=0.5, extent=[0, length, 0, 1])

    # Generate a random matrix for testing and display it on the right
    random_matrix = np.random.rand(10, 10)  # Example matrix with 10 columns
    random_matrix = random_matrix>0.5
    # ax[i, 1].imshow(random_matrix, aspect='auto', cmap='gray')
    ax[i, 1].imshow(random_matrix, cmap='Blues')
    # ax[i, 1].axis('off')  # Turn off the axis for the matrix plot
    ax[i, 1].set_xticks([])  # Turn off the x-ticks for the matrix plot
    ax[i, 1].set_yticks([])  # Turn off the y-ticks for the matrix plot

plt.tight_layout()
plt.savefig('completeness_analysis/selected_channels.png')