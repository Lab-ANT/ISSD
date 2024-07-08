from issd import ISSD
from miniutils import *
import time
import os

# fname_list = os.listdir('data/MoCap/raw')
# fname_list.sort()
# fname_list = fname_list[:len(fname_list)//2]
# datalist = []
# state_seq_list = []
# for fname in fname_list:
#     data, state_seq = load_data(f'data/MoCap/raw/{fname}')
#     datalist.append(data)
#     state_seq_list.append(state_seq)
# for i in range(1,11):
#     start = time.time()
#     selector = ISSD(n_jobs=i)
#     selector.compute_matrices(datalist, state_seq_list)
#     selected_channels_qf = selector.get_qf_solution(4)
#     selected_channels_cf = selector.get_cf_solution(4)
#     end = time.time()
#     print(f'time taken for {i} iteration: {end-start} seconds')

# example_data, state_seq = load_data('data/MoCap/raw/86_02.npy')
# time_consumed = []
# for i in range(1,11):
#     selector = ISSD(n_jobs=i)
#     start_nn = time.time()
#     selector.compute_matrices([example_data], [state_seq])
#     end_nn = time.time()
#     start_qf = time.time()
#     selected_channels_qf = selector.get_qf_solution(4)
#     end_qf = time.time()
#     start_cf = time.time()
#     selected_channels_cf = selector.get_cf_solution(4)
#     end_cf = time.time()
#     time_consumed.append([end_nn-start_nn, end_qf-start_qf, end_cf-start_cf])
#     print(f'time taken for {i} iteration: {end_cf-start_nn} seconds')
# time_consumed = np.array(time_consumed)
# np.save('time_consumed.npy', time_consumed)

time_consumed = np.load('time_consumed.npy')
print(time_consumed)
# time_consumed[:,1] = time_consumed[:,1]*1000
# time_consumed[:,2] = time_consumed[:,2]*1000
import matplotlib.pyplot as plt
# plot stacked line chart
plt.style.use('classic')
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(range(1,11), time_consumed[:,0], label='nntest', marker='o')
ax.plot(range(1,11), time_consumed[:,1], label='QF searching', marker='^')
ax.plot(range(1,11), time_consumed[:,2], label='CF searching', marker='s')
# plt.stackplot(range(1,11), time_consumed.T, labels=['NN', 'QF', 'CF'])
# log the y-axis
ax.set_yscale('log')
ax.set_xlabel('Number of threads')
ax.set_ylabel('Computation Time (s)')
# ax.set_title('Time taken for different number of threads')
# zoom in the area of interest, using a zoomin box
# x: 8-10
# y: 0-0.01
# axins = ax.inset_axes([0.5, 0.1, 0.4, 0.4])
# # axins.plot(range(1,11), time_consumed[:,0], label='NN')
# # axins.plot(range(1,11), time_consumed[:,1], label='QF')
# # axins.plot(range(1,11), time_consumed[:,2], label='CF')
# axins.stackplot(range(1,11), time_consumed.T, labels=['NN', 'QF', 'CF'])
# axins.set_xlim(9.99, 10)
# y = time_consumed[-1,0]
# print(y)
# axins.set_ylim(y, y+0.01)
# ax.indicate_inset_zoom(axins)
# show the legend
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=14.2)
ax.legend()
plt.tight_layout()
plt.savefig('time_taken.png')
plt.savefig('time_taken.pdf')