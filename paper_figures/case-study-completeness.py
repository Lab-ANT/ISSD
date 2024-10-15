import numpy as np
import matplotlib.pyplot as plt

"""
Plot all
"""
# datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']

# for d in datasets:
#     completeness = np.load(f'completeness_analysis/{d}_completeness.npy')
#     quality = np.load(f'completeness_analysis/{d}_quality.npy')

#     plt.style.use('classic')
#     plt.figure(figsize=(10, 2))
#     plt.imshow(completeness, cmap='viridis', interpolation='nearest')
#     plt.colorbar()
#     plt.tight_layout()
#     plt.savefig(f'completeness_analysis/fig_{d}_completeness.png')
#     plt.close()

#     plt.style.use('classic')
#     plt.figure(figsize=(10, 2))
#     plt.imshow(quality, cmap='viridis', interpolation='nearest')
#     plt.colorbar()
#     plt.tight_layout()
#     plt.savefig(f'completeness_analysis/fig_{d}_quality.png')
#     plt.close()

"""
Plot MoCap
"""
completeness = np.load('completeness_analysis/MoCap_completeness.npy')
quality = np.load('completeness_analysis/MoCap_quality.npy')

# plt.style.use('classic')
# plt.figure(figsize=(10, 2))
# plt.imshow(completeness, cmap='viridis', interpolation='nearest')

# plt.colorbar()
# plt.tight_layout()
# plt.savefig('completeness_analysis/imshow_completeness.png')
# plt.close()

import matplotlib.pyplot as plt
import numpy as np

# # Creating the plot with imshow
fig, ax = plt.subplots(figsize=(5, 1.5))

# Plotting the imshow with rectangular data
cax = ax.imshow(completeness, aspect='auto', cmap='viridis')
# ax.set_xlabel('Channel Index', fontsize=12)
ax.set_ylabel('Instance Index', fontsize=10)
ax.set_yticks(np.arange(0, completeness.shape[0], 2),[0,2,4,6,8])

# Adding a colorbar and manually adjusting to match the height exactly
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax_divider = divider.append_axes("right", size="2%", pad=0.1)

# Adding the colorbar to the side plot and setting the label
cbar = fig.colorbar(cax, cax=cax_divider)
# set the ticks of the colorbar, just use 'high' and 'low' to keep it simple
cbar.set_ticks([0, 1])
# cbar.set_ticklabels(['Low', 'High'])
cbar.set_label('Completeness')

plt.tight_layout()
plt.savefig('completeness_analysis/imshow_completeness.png')
plt.close()

# # Creating the plot with imshow
fig, ax = plt.subplots(figsize=(5, 1.5))

# Plotting the imshow with rectangular data
cax = ax.imshow(completeness, aspect='auto', cmap='viridis')
# ax.set_xlabel('Channel Index', fontsize=12)
ax.set_ylabel('Instance Index', fontsize=10)
ax.set_yticks(np.arange(0, quality.shape[0], 2),[0,2,4,6,8])

# Adding a colorbar and manually adjusting to match the height exactly
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax_divider = divider.append_axes("right", size="2%", pad=0.1)

# Adding the colorbar to the side plot and setting the label
cbar = fig.colorbar(cax, cax=cax_divider)
# set the ticks of the colorbar, just use 'high' and 'low' to keep it simple
cbar.set_ticks([0, 1])
# cbar.set_ticklabels(['Low', 'High'])
cbar.set_label('Quality')

plt.tight_layout()
plt.savefig('completeness_analysis/imshow_quality.png')
plt.close()
