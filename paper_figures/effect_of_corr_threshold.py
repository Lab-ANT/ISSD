import matplotlib.pyplot as plt

plt.style.use('classic')
plt.figure(figsize=(4, 3))

plt.plot([1,2,3,4,5], label='MoCap')
plt.plot([5,4,3,2,1], label='ActRecTut')
plt.plot([2,3,4,5,6], label='PAMAP2')
plt.plot([3,4,5,6,7], label='SynSeg')
plt.plot([4,5,6,7,8], label='USC-HAD')

plt.legend()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=3)
plt.legend(loc='upper center', ncol=3, fontsize=8)
plt.savefig('output/dataset_analysis/figs/effect_corr.png')
plt.savefig('output/dataset_analysis/figs/effect_corr.pdf')
plt.close()