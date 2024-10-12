import matplotlib.pyplot as plt
import numpy as np

color=['#c9393e', '#497fc0', '#29517c', '#9694e7', '#ecd268', '#9dc37d', '#ddd2a4', '#00B4B7', '#008F9D', '#916142']

with open('corr.out') as f:
    lines = f.readlines()
    lines = [lines.split(' ') for lines in lines]
    lines = [float(line[3]) for line in lines]
    print(lines)
    num = 40
    # split into 5 datasets
    nmi = [lines[i:i+num] for i in range(0, len(lines), num)]
    nmi = np.array(nmi)
    print(nmi.shape)
    nmi = np.mean(nmi, axis=0)
    num = 8
    nmi = [nmi[i:i+num] for i in range(0, len(nmi), num)]
    nmi = np.array(nmi)
    print(nmi.shape)

# dmthods = ['Time2State', 'E2USD', 'TICC', 'AutoPlait']
datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'SynSeg', 'USC-HAD']

plt.style.use('classic')
plt.figure(figsize=(4,3))
x = np.arange(8)
y_min = np.min(nmi)
y_max = np.max(nmi)
# print(x)
for i in range(5):
    plt.plot(x, nmi[i], lw=2, marker='o', label=datasets[i], color=color[i+1])
nmi = np.mean(nmi, axis=0)
plt.plot(x, nmi, lw=2, marker='o', label='Average', color=color[0])
plt.ylim(y_min*0.8, y_max*1.2)
plt.yticks([0.5, 0.6, 0.7, 0.8])
plt.ylabel('NMI')
plt.xlabel('Pearson Correlation Threshold')
plt.xticks(x, ['0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.6), ncol=2, fontsize=11)
plt.tight_layout()
plt.savefig('archived_output/figs/effect_of_corr_threshold.png')
plt.savefig('archived_output/figs/effect_of_corr_threshold.pdf')