import sys
sys.path.append('.')
import os
from miniutils import load_data, find_cut_points_from_state_seq, reorder_label, compact_state_seq, reorder_state_seq
import matplotlib.pyplot as plt
import sys
import matplotlib
sys.path.append('.')

os.makedirs('completeness_analysis', exist_ok=True)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

fname = 'data/MoCap/raw/86_08.npy'
data, state_seq = load_data(fname)
# data = data[::10]
# state_seq = state_seq[::10]
state_seq = state_seq.reshape(-1, 1)
selected_channels = [58, 28, 38, 55]
# c = [268, 278, 284, 284]
gray_position = [
    [2,3,4,7,8,9],
    [0,4,6,7,10],
    [0,4,7,10],
    [3,4,7]
]

state_seq = reorder_state_seq(state_seq).reshape(-1, 1)
# state_seq = reorder_label(state_seq)
cps = find_cut_points_from_state_seq(state_seq)
length = data.shape[0]
compacted_state_seq = compact_state_seq(state_seq)
print(len(compacted_state_seq), compacted_state_seq)
print(len(cps), cps)
from sklearn.preprocessing import StandardScaler
results = []
channel_set = data[:, selected_channels]
channel_set = StandardScaler().fit_transform(channel_set)
# # scale to 0-1
channel_set = (channel_set - channel_set.min()) / (channel_set.max() - channel_set.min())
channel_set = [channel_set[:, i] for i in range(channel_set.shape[1])]

plt.style.use('classic')
fig, ax = plt.subplots(nrows=5, figsize=(8, 5))
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
    ax[i].imshow(temp_state_seq.T, aspect='auto', alpha=0.6, cmap='tab20', extent=[0, length, 0, 1])

    for k in range(len(gray_position[i])):  
        start, end = cps[gray_position[i][k]], cps[gray_position[i][k]+1]  
        rect = plt.Rectangle((start, 0), end-start, 1, facecolor='white', alpha=1, edgecolor='white')  
        ax[i].add_patch(rect) 

    ax[i].plot(channel, lw=1.5)
    results.append(idx)
    string = ', '.join([str(r) for r in selected_channels[:i+1]])
    # title_left = f'Setp {i+1} selected: {idx}.'+' Result set: {'+string+'}, current completeness='+str(c[i])
    title_left = f'Setp {i+1} selected: {idx}.'+' Result set: {'+string+'}'
    # title_right = 'Current completeness='+str(c[i])
    ax[i].set_title(title_left, fontsize=12, loc='left') # fontweight='bold'
    # ax[i].set_title(title_right, fontsize=12, loc='right')
    for cp in cps[:-1]:
        ax[i].axvline(cp, color='black', linestyle='--')
    ax[i].set_xlim(0, length)
    # remove x ticks for the first 3 ax
    # if i < len(selected_channels):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
# plot all channels in the last ax
ax[-1].imshow(state_seq.T, aspect='auto', alpha=0.6, cmap='tab20', extent=[0, length, 0, 1])
for cp in cps[:-1]:
    ax[-1].axvline(cp, color='black', linestyle='--')
for channel in channel_set:
    ax[-1].plot(channel, lw=1.5)
ax[-1].set_yticks([])
ax[-1].set_xlim(0, length)
ax[-1].set_title('Final selection', fontsize=12, loc='left')

plt.tight_layout()

# compress the file size of the saved figure
plt.savefig('archive/figs/selected_channels.png', dpi=50)
# plt.savefig('archive/figs/selected_channels.pdf', dpi=50)