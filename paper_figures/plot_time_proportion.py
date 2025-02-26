import numpy as np
import matplotlib.pyplot as plt
import json

data_json = json.load(open('archive/other_output/time_consumed.json', 'r'))

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(5,5))

species = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']

width = 0.2
# bottom = np.zeros(len(species))
offset = -0.3
x = np.array([1,2,3,4,5])
colors = color=['#ecd268', '#c9393e', '#9694e7', '#497fc0']
i = 0
for boolean, weight_count in data_json.items():
    # ax.bar(x+offset, weight_count, width, label=boolean, color=colors[i], edgecolor=colors[i])
    ax.bar(x+offset, weight_count, width, label=boolean, color=colors[i])
    offset += width
    i+=1
ax.set_xticks(x)
ax.set_xticklabels(species, size=11.5)

ax.set_yscale('log')

ax.set_xlim(0.3, 5.5)
ax.set_ylim(1e-4, 1e4)
ax.set_xlabel('Datasets', fontsize=16)
ax.set_ylabel('Computation Time (s)', fontsize=16)
ax.set_title('Computation Time on Datasets', fontsize=16)
# set ticks size larger
ax.tick_params(axis='y', which='major', length=8, width=1, labelright=False)
ax.tick_params(axis='y', which='minor', length=4, width=1, labelright=False)
ax.tick_params(axis='y', labelsize=14)
# hide right y-axis
ax.yaxis.set_tick_params(right=False)
# hide right ticks
ax.yaxis.set_ticks_position('left')
plt.legend(fontsize=13, loc='upper center', ncol=2)
plt.tight_layout()
plt.savefig('archive/figs/time_datasets.png')
# plt.savefig('archive/figs/time_datasets.pdf')