import numpy as np
import matplotlib.pyplot as plt
import json
import os

os.makedirs('output/figs', exist_ok=True)

data_json = json.load(open('time_comparison.json', 'r'))

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(10,4))

species = ['MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'SynSeg']

width = 0.1
# bottom = np.zeros(len(species))
offset = -0.3
x = np.array([1,2,3,4,5])
colors=['#c9393e', '#497fc0', '#29517c', '#9694e7', '#ecd268', '#9dc37d', '#ddd2a4', '#00B4B7', '#008F9D', '#916142']

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
ax.set_ylim(1e-4, 1e3)
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
plt.legend(fontsize=13, loc='upper center', ncol=4)
plt.tight_layout()
plt.savefig('output/figs/time_comparison.png')
plt.savefig('output/figs/time_comparison.pdf')