import pandas as pd
import numpy as np
import os

labeled_path = 'data/raw/MoCap-new/'
raw_path = 'data/raw/MoCap-csv/'
f_list = os.listdir(labeled_path)
print(f_list)

os.makedirs(f'data/MoCap/raw', exist_ok=True)

for fname in f_list:
    print(fname)
    df = pd.read_csv(os.path.join(labeled_path, fname))
    df_raw = pd.read_csv(os.path.join(raw_path, fname[:-12]+'.csv'))
    label = df['label']
    data = df_raw.to_numpy()
    data = np.vstack((data.T, label)).T
    print(data.shape)
    np.save(f'data/MoCap/raw/{fname[:-12]}.npy', data)