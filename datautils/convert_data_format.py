"""
Created by Chengyu.
Convert data format for all datasets for easy handling.
"""

import scipy
import os
import numpy as np

def seg_to_label(label):
    pre = 0
    seg = []
    for l in label:
        seg.append(np.ones(l-pre,dtype=int)*label[l])
        pre = l
    result = np.concatenate(seg)
    return result

def load_USC_HAD(subject, target, dataset_path):
    prefix = os.path.join(dataset_path,'USC-HAD/Subject'+str(subject)+'/')
    fname_prefix = 'a'
    fname_postfix = 't'+str(target)+'.mat'
    data_list = []
    label_json = {}
    total_length = 0
    for i in range(1,13):
        data = scipy.io.loadmat(prefix+fname_prefix+str(i)+fname_postfix)
        data = data['sensor_readings']
        data_list.append(data)
        total_length += len(data)
        label_json[total_length]=i
    label = seg_to_label(label_json)
    return np.vstack(data_list), label

def fill_nan(data):
    """
    Pad nan with the previous value
    Note there may be a continuous nan sequence
    """
    idx_nan = np.argwhere(np.isnan(data))
    for idx in idx_nan:
        if idx[0] == 0:
            data[idx[0],idx[1]] = 0
        else:
            data[idx[0],idx[1]] = data[idx[0]-1,idx[1]]
    return data

def load_PAMAP2(path):
    data = np.loadtxt(path)
    state_seq = data[:,1].astype(int)
    data = data[:,2:]
    data = fill_nan(data)
    return data, state_seq

# USC-HAD
print("processing USC-HAD ...")
data_path = 'data/raw'
os.makedirs('data/USC-HAD/raw', exist_ok=True)
for s in range(1,15):
    data, label = load_USC_HAD(s, 1, data_path)
    # 2x downsampling
    data = np.concatenate((data, label.reshape(-1,1)), axis=1)[::2]
    np.save(os.path.join(f'data/USC-HAD/raw/s{s}.npy'), data)

# ActRecTut
print("processing ActRecTut")
data_path = 'data/raw'
os.makedirs('data/ActRecTut/raw', exist_ok=True)
for i in range(1,3):
    dataset_path = f'data/raw/ActRecTut/subject{i}_walk/data.mat'
    mat = scipy.io.loadmat(dataset_path)
    data = mat['data']
    labels = mat['labels']
    # cut data and label into two parts
    data = np.concatenate((data, labels), axis=1)
    np.save(f'data/ActRecTut/raw/subject{i}_walk.npy', data)

# PAMAP2
print("processing PAMAP2")
data_path = 'data/raw/PAMAP2/Protocol'
os.makedirs('data/PAMAP2/raw', exist_ok=True)
f_list = os.listdir(data_path)
for fname in f_list:
    # skip subject 109, which is too short under 5x down sampling
    if fname == 'subject109.dat':
        continue
    data, state_seq = load_PAMAP2(os.path.join(data_path, fname))
    data = np.concatenate((data, state_seq.reshape(-1,1)), axis=1)[::5]
    np.save(f'data/PAMAP2/raw/{fname[:-4]}.npy', data)