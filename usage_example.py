"""
Created by Chengyu on 2024/3/10.
This is an example of how to use the ISSD and IRSD.
"""

import numpy as np
from issd import issd, irsd
from miniutils import *

# EXAMPLE OF ISSD, select on one time series
example_data, state_seq = load_data('data/MoCap/raw/86_02.npy')
# quality first strategy
result = issd(example_data, state_seq, 4, strategy='bf', save_path='output/example-issd-bf', n_jobs=10)
print(f'ISSD-BF result: {result}, The intermediate results are saved in output/example-issd-bf folder.')
# completeness first strategy
result = issd(example_data, state_seq, 4, strategy='cf', save_path='output/example-issd-cf', n_jobs=10)
print(f'ISSD-CF result: {result}, The intermediate results are saved in output/example-issd-cf folder.')
