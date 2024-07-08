"""
This script generates the example figure in the paper and should not be published.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from TSpy.label import seg_to_label

# 2-5, 2-9
# 1-6, 23

channel_list = []

# the derimental channel
df = pd.read_csv(f'../data/SMD/train/machine-{1}-{6}.txt')
df2 = pd.read_csv(f'../data/SMD/test/machine-{1}-{6}.txt')
data = np.concatenate((df.to_numpy(), df2.to_numpy()), axis=0)
channel_list.append(data[:,23])

# the non-informative channel
df = pd.read_csv(f'../data/SMD/train/machine-{3}-{4}.txt')
df2 = pd.read_csv(f'../data/SMD/test/machine-{3}-{4}.txt')
data = np.concatenate((df.to_numpy(), df2.to_numpy()), axis=0)
channel_list.append(data[:,31])

# other channels
df = pd.read_csv(f'../data/SMD/train/machine-{1}-{1}.txt')
df2 = pd.read_csv(f'../data/SMD/test/machine-{1}-{1}.txt')
data = np.concatenate((df.to_numpy(), df2.to_numpy()), axis=0)
channel_list.append(data[:,3])
channel_list.append(data[:,1])
channel_list.append(data[:,22])

# convert to hex
blue = '#%02x%02x%02x'%(62, 119, 179)
print(blue)

# state_sequences = []
# seg = {3000:1,30000:2,56956:1}
# print(seg_to_label(seg), len(seg_to_label(seg)))

plt.style.use('classic')
fig, ax = plt.subplots(nrows=len(channel_list)+1, figsize=(7,3))
k=1
for i in range(len(channel_list)):
    # ax[i].imshow(np.vstack([np.array(seg_to_label(seg)).reshape(1,-1),
    #                        np.array(seg_to_label(seg)).reshape(1,-1)]),
    #              aspect='auto',
    #              interpolation='nearest',
    #              cmap='Set3',
    #              alpha=0.5)
    ax[i].plot(channel_list[i][::10], lw=1.5, color=blue)
    # ax[i].set_title(f'channel {channel_indices[i]}')
    # place title at the left side of the plot
    # ax[i].text(-0.1, 0.5, f'channel {k}',
    #            fontsize=10,
    #            horizontalalignment='center',
    #            verticalalignment='center',
    #            transform=ax[i].transAxes)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_ylim(-0.2,1.2)
    k+=1

ax[5].set_xticks([])
ax[5].set_yticks([])
plt.tight_layout()
plt.savefig('paper_figures/example.pdf')