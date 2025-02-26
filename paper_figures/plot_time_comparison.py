import numpy as np
import matplotlib.pyplot as plt
import json
import os

result_json_list = [json.load(open(f'archive/other_output/comparison_execution{i}.json', 'r')) for i in range(5)]

methods = ['issd', 'pca', 'lda', 'umap', 'sfm', 'ecp', 'ecs']
datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'SynSeg', 'USC-HAD']
species = ['ISSD', 'PCA', 'LDA', 'UMAP', 'SFM', 'ECP', 'ECS']

avg_table = []
for result_json in result_json_list:
    table = []
    for m in methods:
        m_list = []
        for d in datasets:
            m_list.append(result_json[f'{m}_{d}'])
        table.append(m_list)
    avg_table.append(table)
avg_table = np.array(avg_table)
print(avg_table.shape)

avg_table = np.mean(avg_table, axis=0)
print(avg_table.shape)

os.makedirs('archive/figs', exist_ok=True)

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(10,4))

width = 0.1
# # bottom = np.zeros(len(species))
offset = -0.3
x = np.array([1,2,3,4,5])
colors=['#c9393e', '#497fc0', '#29517c', '#9694e7', '#ecd268', '#9dc37d', '#ddd2a4', '#00B4B7', '#008F9D', '#916142']

num_cols = avg_table.shape[1]
num_rows = avg_table.shape[0]

for i in range(num_rows):
    print(avg_table[i,:].shape)
    ax.bar(x+offset, avg_table[i,:], width, label=species[i], color=colors[i])
    offset += width

# annotate time consumption for issd,
# the first bar in each group
for i in range(num_cols):
    ax.text(x[i]-0.45, avg_table[0,i]*1.2, f'{avg_table[0,i]:.2f}s', fontsize=12, color='black')

ax.set_xticks(x)
ax.set_xticklabels(datasets, size=14)
ax.set_yscale('log')
ax.set_xlim(0.5, 5.5)
ax.set_ylim(1e-4, 1e3)
ax.set_xlabel('Datasets', fontsize=16)
ax.set_ylabel('Computation Time (s)', fontsize=16)
# ax.set_title('Computation Time on Datasets', fontsize=16)
# set ticks size larger
ax.tick_params(axis='y', which='major', length=8, width=1, labelright=False)
ax.tick_params(axis='y', which='minor', length=4, width=1, labelright=False)
ax.tick_params(axis='y', labelsize=14)
# hide right y-axis
ax.yaxis.set_tick_params(right=False)
# hide right ticks
ax.yaxis.set_ticks_position('left')
plt.legend(fontsize=13, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.2))
plt.tight_layout()
plt.savefig('archive/figs/time_comparison.png')
# plt.savefig('archive/figs/time_comparison.pdf')