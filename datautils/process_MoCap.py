import pandas as pd
import numpy as np
import os

raw_path = 'data/raw/'
f_list = os.listdir(raw_path)
print(f_list)

os.makedirs('data/MoCap/raw', exist_ok=True)

# for fname in f_list:
#     df = pd.read_csv(os.path.join(raw_path, fname), header=None)
#     df_raw = pd.read_csv(os.path.join(raw_path, fname), header=None)
#     print(df.shape, df_raw.shape)
#     # print(df_raw)
#     label = df.to_numpy()[:,4]
#     data = df_raw.to_numpy()
#     print(data.shape, label.shape)
#     data = np.vstack((data.T, label)).T
#     print(data.shape)
#     np.save(os.path.join(script_path, '../data/processed/MoCap', fname[4:-4]+'.npy'), data)