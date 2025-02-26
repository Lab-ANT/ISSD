import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('archive/figs', exist_ok=True)

data = np.load('archive/other_output/effect_of_core_num.npy')
print(data.shape)

# calculate acceleration
acceleration = data[0] / data

# calculate efficiency
efficiency = acceleration / np.arange(1, 11)

print(efficiency)

x = np.arange(1, 11)

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(4.5, 4.5))

# plot computation time and acceleration separately
# computation time uses the left y-axis
# acceleration and efficiency uses the right y-axis

line1, = ax.plot(x, data, marker='o', color='blue', lw=2, label='Computation Time (s)')
ax.set_xlabel('Number of Cores', fontsize=16)
ax.set_ylabel('Computation Time (s)', fontsize=16)

ax2 = ax.twinx()
line2, = ax2.plot(x, acceleration, marker='s', lw=2, color='red', label='Speedup')
line3, = ax2.plot(x, efficiency, marker='^', lw=2, color='green', label='Efficiency')
ax2.set_ylim([0, 11])
ax2.set_ylabel('Speedup & Efficiency', fontsize=16)

# Create a legend box
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
legend = ax.legend(lines, labels, loc='upper center', frameon=True, fontsize=13)

plt.xlim([1, 10])
plt.tight_layout()
plt.savefig('archive/figs/effect_of_core_num.png')
# plt.savefig('archive/figs/effect_of_core_num.pdf')