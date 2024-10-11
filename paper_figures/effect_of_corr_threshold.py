import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import normalized_mutual_info_score

datasets = ['MoCap', 'ActRecTut', 'PAMAP2', 'SynSeg', 'USC-HAD']
dmthods = ['time2state', 'e2usd', 'ticc', 'autoplait']

result_json = {}
# for corr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
for m in ['time2state', 'e2usd']:
    for d in datasets:
        for corr in [0.2, 0.3]:
            nmi_list = []
            for i in range(1, 6):
                path = f'archive/corr_effect/{d}/{corr}/{m}{i}/issd/'
                fname_list = os.listdir(path)
                nmi_list_on_dataset = []
                for fname in fname_list:
                    result = np.load(path + fname)
                    nmi = normalized_mutual_info_score(result[0, :], result[1, :])
                    # print(nmi)
                    nmi_list_on_dataset.append(nmi)
                nmi_on_dataset = np.mean(nmi_list_on_dataset)
                nmi_list.append(nmi_on_dataset)
            mean_nmi = np.mean(nmi_list)
            print(m, d, corr, mean_nmi)
# plt.style.use('classic')
# plt.figure(figsize=(4, 3))

# plt.plot([1,2,3,4,5], label='MoCap')
# plt.plot([5,4,3,2,1], label='ActRecTut')
# plt.plot([2,3,4,5,6], label='PAMAP2')
# plt.plot([3,4,5,6,7], label='SynSeg')
# plt.plot([4,5,6,7,8], label='USC-HAD')

# plt.legend()
# # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=3)
# plt.legend(loc='upper center', ncol=3, fontsize=8)
# plt.savefig('output/dataset_analysis/figs/effect_corr.png')
# plt.savefig('output/dataset_analysis/figs/effect_corr.pdf')
# plt.close()