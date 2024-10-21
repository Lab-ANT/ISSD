import sys
sys.path.append('.')
import os
import numpy as np
from miniutils import load_data, find_cut_points_from_state_seq, reorder_label, compact_state_seq, reorder_state_seq
from issd import ISSD
import matplotlib.pyplot as plt
import sys
import matplotlib
sys.path.append('.')

os.makedirs('completeness_analysis', exist_ok=True)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', ...]

fname = 'data/MoCap/raw/86_08.npy'
data, state_seq = load_data(fname)
state_seq = state_seq.reshape(-1, 1)
# selected_channels = [58, 28, 38, 55]
c = [268, 278, 284, 284]
gray_position = [
    [2,3,4,7,8,9],
    [0,4,6,7,10],
    [0,4,7,10],
    [3,4,6,7]
]

state_seq = reorder_state_seq(state_seq).reshape(-1, 1)
# state_seq = reorder_label(state_seq)
cps = find_cut_points_from_state_seq(state_seq)
length = data.shape[0]
compacted_state_seq = compact_state_seq(state_seq)
print(len(compacted_state_seq), compacted_state_seq)
print(len(cps), cps)
from sklearn.preprocessing import StandardScaler
# plt.style.use('classic')
results = []
channel_set = data[:, selected_channels]
channel_set = StandardScaler().fit_transform(channel_set)
# # scale to 0-1
channel_set = (channel_set - channel_set.min()) / (channel_set.max() - channel_set.min())
channel_set = [channel_set[:, i] for i in range(channel_set.shape[1])]

fig, ax = plt.subplots(nrows=4, figsize=(8, 4))
for i, idx in enumerate(selected_channels):
    channel = channel_set[i]
    # annotate the state index in the first ax
    if i == 0:
        for j in range(len(compacted_state_seq)):
            ax[i].text(cps[j]+100, 
                       0.6, 
                       'S '+str(compacted_state_seq[j]), 
                       fontsize=10,
                       color='black',
                       weight='bold',)
    # corlor the position of gray_position in gray
    temp_state_seq = state_seq.copy()
    ax[i].imshow(temp_state_seq.T, aspect='auto', alpha=0.8, cmap='tab10', extent=[0, length, 0, 1])

    for k in range(len(gray_position[i])):  
        start, end = cps[gray_position[i][k]], cps[gray_position[i][k]+1]  
        rect = plt.Rectangle((start, 0), end-start, 1, facecolor='white', alpha=1)  
        ax[i].add_patch(rect) 

    ax[i].plot(channel)
    results.append(idx)
    string = ', '.join([str(r) for r in selected_channels[:i+1]])
    title = 'Result set: {'+string+'}, completeness='+str(c[i])
    ax[i].set_title(title, fontsize=12, loc='left') # fontweight='bold'
    for cp in cps[:-1]:
        ax[i].axvline(cp, color='black', linestyle='--')
    ax[i].set_xlim(0, length)
    # remove x ticks for the first 3 ax
    if i < len(selected_channels)-1:
        ax[i].set_xticks([])

plt.tight_layout()
plt.savefig('completeness_analysis/selected_channels.png')
plt.savefig('completeness_analysis/selected_channels.pdf')

# import sys 
# sys.path.append('.')
# import os
# import numpy as np
# from miniutils import load_data, find_cut_points_from_state_seq, reorder_label, compact_state_seq, reorder_state_seq
# from issd import ISSD
# import matplotlib.pyplot as plt
# import sys
# import matplotlib
# from matplotlib.colors import ListedColormap

# sys.path.append('.')

# os.makedirs('completeness_analysis', exist_ok=True)

# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', ...]

# fname = 'data/MoCap/raw/86_08.npy'
# data, state_seq = load_data(fname)
# state_seq = state_seq.reshape(-1, 1)
# selected_channels = [58, 28, 38, 55]
# c = [268, 278, 284, 284]
# gray_position = [
#     [2, 3, 4, 7, 8, 9],
#     [0, 4, 6, 7, 10],
#     [0, 4, 7, 10],
#     [3, 4]
# ]

# state_seq = reorder_state_seq(state_seq).reshape(-1, 1)
# cps = find_cut_points_from_state_seq(state_seq)
# length = data.shape[0]
# compacted_state_seq = compact_state_seq(state_seq)

# from sklearn.preprocessing import StandardScaler

# results = []
# channel_set = data[:, selected_channels]
# channel_set = StandardScaler().fit_transform(channel_set)
# channel_set = (channel_set - channel_set.min()) / (channel_set.max() - channel_set.min())
# channel_set = [channel_set[:, i] for i in range(channel_set.shape[1])]

# # Custom colormap: Use gray for -1, and 'tab10' for other states
# from matplotlib.colors import ListedColormap

# # Convert the tuple to a list
# cmap_colors = list(plt.cm.tab10.colors)
# # Add a specific gray color for the -1 value (which represents gray regions)
# cmap = ListedColormap(['white'] + cmap_colors)

# fig, ax = plt.subplots(nrows=4, figsize=(8, 4))
# for i, idx in enumerate(selected_channels):
#     channel = channel_set[i]
#     # Annotate the state index in the first ax
#     if i == 0:
#         for j in range(len(compacted_state_seq)):
#             ax[i].text(cps[j] + 50,
#                        0.6,
#                        'S ' + str(compacted_state_seq[j]),
#                        fontsize=10,
#                        color='black',
#                        weight='bold',)

#     # Create a copy of the state sequence
#     temp_state_seq = state_seq.copy()

#     # Mark the positions to be gray (-1) in the temporary state sequence
#     for k in range(len(gray_position[i])):
#         temp_state_seq[cps[gray_position[i][k]]:cps[gray_position[i][k] + 1]] = -1

#     # Ensure other values are within a valid range (positive integer) for color mapping
#     temp_state_seq[temp_state_seq >= 0] = temp_state_seq[temp_state_seq >= 0] + 1

#     # Display the state sequence with gray regions
#     ax[i].imshow(temp_state_seq.T, aspect='auto', alpha=0.8, cmap=cmap, extent=[0, length, 0, 1])

#     # Plot the channel data
#     ax[i].plot(channel)
#     results.append(idx)
#     string = ', '.join([str(r) for r in selected_channels[:i + 1]])
#     title = 'Result set: {' + string + '}, completeness=' + str(c[i])
#     ax[i].set_title(title, fontsize=12, loc='left')

#     # Add vertical lines for cut points
#     for cp in cps[:-1]:
#         ax[i].axvline(cp, color='black', linestyle='--')

#     ax[i].set_xlim(0, length)

#     # Remove x ticks for the first 3 axes
#     if i < len(selected_channels) - 1:
#         ax[i].set_xticks([])

# plt.tight_layout()
# plt.savefig('completeness_analysis/selected_channels.png')
# plt.savefig('completeness_analysis/selected_channels.pdf')

# import sys
# sys.path.append('.')
# import os
# import numpy as np
# from miniutils import load_data, find_cut_points_from_state_seq, reorder_label, compact_state_seq, reorder_state_seq
# from issd import ISSD
# import matplotlib.pyplot as plt
# import sys
# import matplotlib
# sys.path.append('.')

# os.makedirs('completeness_analysis', exist_ok=True)

# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', ...]

# fname = 'data/MoCap/raw/86_08.npy'
# data, state_seq = load_data(fname)
# state_seq = state_seq.reshape(-1, 1)
# selected_channels = [58, 28, 38, 55]
# c = [268, 278, 284, 284]
# gray_position = [
#     [2,3,4,7,8,9],
#     [0,4,6,7,10],
#     [0,4,7,10],
#     [3,4,6,7]
# ]

# state_seq = reorder_state_seq(state_seq).reshape(-1, 1)
# # state_seq = reorder_label(state_seq)
# cps = find_cut_points_from_state_seq(state_seq)
# length = data.shape[0]
# compacted_state_seq = compact_state_seq(state_seq)
# print(len(compacted_state_seq), compacted_state_seq)
# print(len(cps), cps)
# from sklearn.preprocessing import StandardScaler
# # plt.style.use('classic')
# results = []
# channel_set = data[:, selected_channels]
# channel_set = StandardScaler().fit_transform(channel_set)
# # # scale to 0-1
# channel_set = (channel_set - channel_set.min()) / (channel_set.max() - channel_set.min())
# channel_set = [channel_set[:, i] for i in range(channel_set.shape[1])]

# fig, ax = plt.subplots(nrows=4, figsize=(8, 4))
# for i, idx in enumerate(selected_channels):
#     channel = channel_set[i]
#     # annotate the state index in the first ax
#     if i == 0:
#         for j in range(len(compacted_state_seq)):
#             ax[i].text(cps[j]+50, 
#                        0.8, 
#                        'S '+str(compacted_state_seq[j]), 
#                        fontsize=10,
#                        color='red',
#                        weight='bold',)
#     # corlor the position of gray_position in gray
#     temp_state_seq = state_seq.copy()
#     for k in range(len(gray_position[i])):
#         temp_state_seq[cps[gray_position[i][k]]:cps[gray_position[i][k]+1]] = -1
#     ax[i].imshow(temp_state_seq.T, aspect='auto', alpha=0.8, cmap='tab10', extent=[0, length, 0, 1])

#     # for k in range(len(gray_position[i])):  
#     #     start, end = cps[gray_position[i][k]], cps[gray_position[i][k]+1]  
#     #     rect = plt.Rectangle((start, 0), end-start, 1, facecolor='gray', alpha=0.5)  
#     #     ax[i].add_patch(rect) 
#     # for k in range(len(gray_position[i])):
#     #     temp_state_seq = state_seq[cps[gray_position[i][k]]:cps[gray_position[i][k]+1]]
#     #     ax[i].imshow(temp_state_seq.T,
#     #                  aspect='auto',
#     #                  alpha=0.5,
#     #                 #  cmap='tab20',
#     #                  extent=[cps[gray_position[i][k]], cps[gray_position[i][k]+1], 0, 1])
#     # for k in range(i):
#     #     ax[i].plot(channel_set[k], color='gray', alpha=0.6)
#     ax[i].plot(channel)
#     results.append(idx)
#     string = ', '.join([str(r) for r in selected_channels[:i+1]])
#     title = 'Result set: {'+string+'}, completeness='+str(c[i])
#     ax[i].set_title(title, fontsize=12, loc='left') # fontweight='bold'
#     for cp in cps[:-1]:
#         ax[i].axvline(cp, color='black', linestyle='--')
#     ax[i].set_xlim(0, length)
#     # remove x ticks for the first 3 ax
#     if i < len(selected_channels)-1:
#         ax[i].set_xticks([])

# plt.tight_layout()
# plt.savefig('completeness_analysis/selected_channels.png')
# plt.savefig('completeness_analysis/selected_channels.pdf')

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
# data, state_seq = load_data(fname)
# selector = ISSD()
# selector.compute_matrices([data], [state_seq])
# # selector.compute_completeness_quality()
# selector.get_cf_solution(4)
# print(selector.cf_solution)
# selected_channels = selector.cf_solution

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

# from sklearn.preprocessing import StandardScaler
# # plt.style.use('classic')
# results = []
# channel_set = data[:, selected_channels]
# channel_set = StandardScaler().fit_transform(channel_set)
# # # scale to 0-1
# channel_set = (channel_set - channel_set.min()) / (channel_set.max() - channel_set.min())
# channel_set = [channel_set[:, i] for i in range(channel_set.shape[1])]

# fig, ax = plt.subplots(nrows=4, ncols=2, gridspec_kw={'width_ratios': [8, 1]}, figsize=(8, 4))
# for i, idx in enumerate(selected_channels):
#     # channel = data[:, idx]
#     # # scale to 0-1
#     # channel = (channel - channel.min()) / (channel.max() - channel.min())
#     # channel_set.append(channel)

#     channel = channel_set[i]

#     # imshow state sequence at the same ax. y range is 0-1
#     ax[i, 0].imshow(state_seq.T, aspect='auto', cmap='tab10', alpha=0.5, extent=[0, length, 0, 1])

#     # plot the former results in the same ax, gray
#     # for k in range(i):
#     #     ax[i, 0].plot(channel_set[k], color='gray', alpha=0.6)
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

# import sys
# sys.path.append('.')
# import os
# import numpy as np
# from miniutils import load_data, find_cut_points_from_state_seq, reorder_label, matrix_OR
# from issd import ISSD
# import matplotlib.pyplot as plt
# import sys
# from sklearn.preprocessing import StandardScaler
# sys.path.append('.')

# os.makedirs('completeness_analysis', exist_ok=True)

# """
# Selection
# """
# fname = 'data/MoCap/raw/86_08.npy'
# # fname = 'data/SynSeg/raw/synthetic2.npy'

# # Select on multiple time series
# data, state_seq = load_data(fname)
# selector = ISSD()
# selector.compute_matrices([data], [state_seq])
# # selector.compute_completeness_quality()
# selector.get_cf_solution(4)
# imatrices = selector.indicator_matrices
# print(selector.cf_solution)
# print(imatrices[0].shape)
# selected_channels = selector.cf_solution
# width = int(np.sqrt(imatrices[0].shape[0]))
# imatrices = [imat.reshape(width, width) for imat in imatrices]
# imatrices = np.array(imatrices)
# print(imatrices.shape)

# """
# Plot
# """
# data, state_seq = load_data(fname)
# state_seq = state_seq.reshape(-1, 1)
# # selected_channels = [58, 59, 48, 56]
# selected_channels = [58, 28, 38, 55]
# state_seq = reorder_label(state_seq)

# cps = find_cut_points_from_state_seq(state_seq)[:-1] # remove the last cp
# length = data.shape[0]

# # plt.style.use('classic')
# results = []
# channel_set = data[:, selected_channels]
# channel_set = StandardScaler().fit_transform(channel_set)
# # scale to 0-1
# channel_set = (channel_set - channel_set.min()) / (channel_set.max() - channel_set.min())
# channel_set = [channel_set[:, i] for i in range(channel_set.shape[1])]
# # imatrices = imatrices[selected_channels]

# # fig, ax = plt.subplots(nrows=4, figsize=(8, 4))
# fig, ax = plt.subplots(nrows=4, ncols=2, gridspec_kw={'width_ratios': [8, 1]}, figsize=(8, 4))
# for i, idx in enumerate(selected_channels):
#     channel = channel_set[i]

#     # plot the former results in the same ax, gray
#     # for k in range(i):
#     #     ax[i, 0].plot(channel_set[k])
#     ax[i, 0].plot(channel)
#     results.append(idx)
#     # ax[i].set_xticks(f'Result set: {results}')

#     for cp in cps:
#         ax[i, 0].axvline(cp, color='black', linestyle='--')
#     ax[i, 0].set_xlim(0, length)
#     # remove x ticks
#     if i < len(selected_channels)-1:
#         ax[i, 0].set_xticks([])

#     # imshow state sequence at the same ax. y range is 0-1
#     ax[i, 0].imshow(state_seq.T, aspect='auto', cmap='tab10', alpha=0.5, extent=[0, length, 0, 1])

#     # Generate a random matrix for testing and display it on the right
#     # random_matrix = np.random.rand(10, 10)  # Example matrix with 10 columns
#     # random_matrix = random_matrix>0.5
#     # ax[i, 1].imshow(random_matrix, aspect='auto', cmap='gray')
#     # ax[i, 1].imshow(random_matrix, cmap='Blues')
#     ax[i, 1].imshow(imatrices[i], cmap='Blues')
#     # ax[i, 1].axis('off')  # Turn off the axis for the matrix plot
#     ax[i, 1].set_xticks([])  # Turn off the x-ticks for the matrix plot
#     ax[i, 1].set_yticks([])  # Turn off the y-ticks for the matrix plot

# plt.tight_layout()
# plt.savefig('completeness_analysis/selected_channels.png')