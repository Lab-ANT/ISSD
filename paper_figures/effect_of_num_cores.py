import numpy as np
import matplotlib.pyplot as plt

data = np.load('effect_of_core_num.npy')

# calculate acceleration
acceleration = data[0] / data

# calculate efficiency
efficiency = acceleration / np.arange(1, 11)

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
ax2.set_ylabel('Speedup & Efficiency', fontsize=16)

# Create a legend box
lines = [line1, line2, line3]
labels = [line.get_label() for line in lines]
legend = ax.legend(lines, labels, loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('effect_of_core_num.png')
plt.savefig('effect_of_core_num.pdf')