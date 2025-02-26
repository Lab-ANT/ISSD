"""
Created by Chengyu.
Used for visualizing the completeness and quality of the MoCap dataset.
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Plot MoCap
"""

completeness = np.load('archive/completeness_analysis/MoCap_completeness.npy')
quality = np.load('archive/completeness_analysis/MoCap_quality.npy')
iter_data = np.load('archive/completeness_analysis/MoCap_iter.npy')
# scale to 0-1
quality = (quality - quality.min()) / (quality.max() - quality.min())

selected_channels = [58, 28, 38, 55]

import matplotlib.pyplot as plt
import numpy as np

# # Creating the plot with imshow
fig, ax = plt.subplots(figsize=(3.8, 0.9))

# Plotting the imshow with rectangular data
# cax = ax.imshow(completeness, aspect='auto', cmap='viridis')
cax = ax.pcolor(completeness, cmap='viridis', edgecolors='k', linewidths=0.5)
# box the corresponding channels
for i in selected_channels:
    ax.add_patch(plt.Rectangle((i, 0), 1, completeness.shape[0], fill=False, edgecolor='red', lw=1))
# ax.set_xlabel('Channel Index', fontsize=12)
ax.set_ylabel('Time Series', fontsize=6)
ax.set_xlabel('Channels', fontsize=6)
ax.set_yticks([])
ax.set_xticks([])
# ax.set_yticks(np.arange(0, completeness.shape[0], 2),[0,2,4,6,8])
# set x tick size
ax.tick_params(axis='x', labelsize=8)

# Adding a colorbar and manually adjusting to match the height exactly
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax_divider = divider.append_axes("right", size="8%", pad=0.1)
# new_im = cax_divider.pcolor(completeness[:,selected_channels], cmap='viridis', edgecolors='k', linewidths=0.5)
new_im = cax_divider.pcolor(iter_data, cmap='viridis', edgecolors='k', linewidths=0.5)
# close x and y ticks
cax_divider.set_yticks([])
cax_divider.set_xticks([])
cax_divider = divider.append_axes("right", size="2%", pad=0.1)

# Adding the colorbar to the side plot and setting the label
cbar = fig.colorbar(cax, cax=cax_divider)
# set the ticks of the colorbar, just use 'high' and 'low' to keep it simple
cbar.set_ticks([0, 1])
# set font size of colorbar
cbar.ax.tick_params(labelsize=8)
# cbar.set_label('Completeness')

plt.tight_layout()
plt.savefig('archive/figs/imshow_completeness.png')
# plt.savefig('completeness_analysis/imshow_completeness.pdf')
plt.close()


# # Creating the plot with imshow
fig, ax = plt.subplots(figsize=(3.8, 0.9))

# Plotting the imshow with rectangular data
# cax = ax.imshow(quality, aspect='auto', cmap='viridis')
cax = ax.pcolor(quality, cmap='viridis', edgecolors='k', linewidths=0.5)
# box the corresponding channels
for i in selected_channels:
    ax.add_patch(plt.Rectangle((i, 0), 1, completeness.shape[0], fill=False, edgecolor='red', lw=1))

# ax.set_xlabel('Channel Index', fontsize=12)
ax.set_ylabel('Time Series', fontsize=6)
ax.set_xlabel('Channels', fontsize=6)
ax.set_yticks([])
ax.set_xticks([])
# ax.set_yticks(np.arange(0, completeness.shape[0], 2),[0,2,4,6,8])
# set x tick size
ax.tick_params(axis='x', labelsize=8)

# Adding a colorbar and manually adjusting to match the height exactly
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax_divider = divider.append_axes("right", size="8%", pad=0.1)
new_im = cax_divider.pcolor(quality[:,selected_channels], cmap='viridis', edgecolors='k', linewidths=0.5)
# close x and y ticks
cax_divider.set_yticks([])
cax_divider.set_xticks([])
cax_divider = divider.append_axes("right", size="2%", pad=0.1)

# Adding the colorbar to the side plot and setting the label
cbar = fig.colorbar(cax, cax=cax_divider)
cbar.set_ticks([0, 1])
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('archive/figs/imshow_quality.png')
# plt.savefig('completeness_analysis/imshow_quality.pdf')
plt.close()