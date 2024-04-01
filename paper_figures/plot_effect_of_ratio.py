import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

plt.style.use('classic')

nmi_list_per_method = []
for method in ['time2state', 'e2usd', 'ticc', 'autoplait']:
    if method in ['time2state', 'e2usd']:
        nmi_list_per_case = []
        for i in range(1,11):
            nmi_list = []
            for j in range(1,6):
                result = np.load(f'output/results/case_study/{method}/case{i}-{j}.npy')
                nmi = normalized_mutual_info_score(result[0], result[1])
                nmi_list.append(nmi)
            nmi_list_per_case.append(np.mean(nmi_list))
        nmi_list_per_method.append(nmi_list_per_case)
    if method in ['ticc', 'autoplait']:
        nmi_list_per_case = []
        for i in range(1,11):
            result = np.load(f'output/results/case_study/{method}/case{i}.npy')
            nmi = normalized_mutual_info_score(result[0], result[1])
            nmi_list_per_case.append(nmi)
        nmi_list_per_method.append(nmi_list_per_case)
print(nmi_list_per_method[0])

plt.style.use('classic')
plt.rcParams['pdf.fonttype'] = 42
plt.figure(figsize=(4.5,4))
plt.plot(nmi_list_per_method[0], label='Time2State', marker='D', lw=3, markersize=8)
plt.plot(nmi_list_per_method[1], label='E2USD', marker='s', lw=3, markersize=8)
plt.plot(nmi_list_per_method[2], label='TICC', marker='o', lw=3, markersize=8)
plt.plot(nmi_list_per_method[3], label='AutoPlait', marker='^', lw=3, markersize=8)
# here are some optional markers: '*', 's', 'o', '^', 'D', 'v', 'x', 'p', 'h'

# add a vertical dashed line at x=5
plt.axvline(x=4, color='gray', linestyle='--', lw=2)
# add a note to the vertical line, use bold font
plt.text(4.1, 0.5, '50%', fontsize=12, fontweight='bold', color='gray')

plt.ylim(0,1.05)
plt.yticks(fontsize=12)
plt.xticks(range(10),[i*10 for i in range(1,11)], fontsize=12)
plt.xlabel('Ratio of Activate Channels (%)', fontsize=16)
plt.ylabel('NMI', fontsize=16)
plt.tight_layout()
# place legend in lower right corner
plt.legend(loc='lower right')
plt.savefig('output/figs/case_study_ratio.pdf')