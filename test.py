import numpy as np
a1 = np.ones(1000)
a2 = np.ones(1000)
a3 = np.ones(10000)

# concatenate a1, a2, a3
a = np.concatenate([a1, a2, a3])
print(a.shape)