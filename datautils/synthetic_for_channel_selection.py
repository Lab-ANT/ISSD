"""
Created by Chengyu.
Synthetic data generator.
The generated data has been provided along with the paper.
If you want to use this script, please install the TSAGen tool
from https://github.com/Lab-ANT/TSAGen
"""

import numpy as np
import os
from shape import RMDF
import tqdm
from TSpy.label import seg_to_label, compact, reorder_label

# configuration
num_relevant_channels = 4
num_irrelevant_channels = 6
num_segs = 15
seg_len = [800, 1200] # 200~1000
num_states = [4, 8] # 4~8
length = 20000
random_state = 2024

# Path configuration
# modify here to change the output path.
script_path = os.path.dirname(__file__)
dataset_name = 'SynSeg'
save_path = os.path.join(script_path, f'../data/{dataset_name}/raw')
os.makedirs(save_path, exist_ok=True)

def gen_seg_json(state_num, seg_len, length):
    seg_num = int(length/seg_len[0])
    # generate state for each segment.
    state_list = np.random.randint(state_num, size=seg_num)
    # print(len(set(state_list)), state_num, state_list)
    while len(set(state_list)) != state_num:
        state_list = np.random.randint(state_num, size=seg_num)
    # generate length for each segment.
    seg_len_list = np.random.randint(low=seg_len[0], high=seg_len[1], size=seg_num)
    seg_json = {}
    total_len = 0
    for i, state, rand_seg_len in zip(range(seg_num), state_list, seg_len_list):
        total_len += rand_seg_len
        if total_len>=length:
            total_len = length
            seg_json[total_len]=state
            break
        seg_json[total_len]=state
    # print(state_num, seg_json)
    return seg_json

# Generate segment json
def generate_seg_json(seg_len, num_states, length, random_state=None):
    # Config seed to generate determinally.
    if random_state is not None:
        np.random.seed(random_state)
    # generate random state num.
    # random_state_num = np.random.randint(low=state_num[0], high=state_num[1]+1)
    random_state_num = np.random.randint(num_states[0], num_states[1]+1)
    # generate state for each segment.
    seg_json = gen_seg_json(random_state_num, seg_len, length)
    state_list = [seg_json[seg] for seg in seg_json]
    while len(set(state_list)) != random_state_num:
        seg_json = gen_seg_json(random_state_num, seg_len, length)
        state_list = [seg_json[seg] for seg in seg_json]
    return seg_json

def gen_channel_from_json(seg_json, forking_depth=1):
    state_list = [seg_json[seg] for seg in seg_json]
    seg_len_list = np.array([seg for seg in seg_json])
    first_seg_len = seg_len_list[0]
    seg_len_list = np.insert(np.diff(seg_len_list), 0, first_seg_len)
    true_state_num = len(set(state_list))
    # This is an object list.
    rmdf_list = [RMDF.RMDF(depth=5) for i in range(true_state_num)]
    for rmdf in rmdf_list:
        rmdf.gen_anchor()
    seg_list = []
    for state, seg_len in zip(state_list, seg_len_list):
        rand_offset = np.random.randint(0, 200)
        seg = [rmdf_list[state].gen(forking_depth=forking_depth, length=200) for i in range(100)]
        # temp = np.concatenate(seg)
        # print(rand_offset, temp.shape, temp[rand_offset:rand_offset+seg_len].shape, temp[:seg_len].shape)
        seg_list.append(np.concatenate(seg)[rand_offset:rand_offset+seg_len])
    result = np.concatenate(seg_list)
    return result

def merge_repeated_states(seg_json):
    """
    This function is generated by Github Copilot.
    Amazing!
    """
    # check if there are ajdecent repeated states.
    state_list = [seg_json[seg] for seg in seg_json]
    flag = False
    for i in range(1, len(state_list)-1):
        if state_list[i] == state_list[i-1] or state_list[i] == state_list[i+1]:
            flag = True
            break
    if not flag:
        return seg_json
    # merge repeated states.
    new_seg_json = {}
    state_list = [seg_json[seg] for seg in seg_json]
    seg_len_list = np.array([seg for seg in seg_json])
    first_seg_len = seg_len_list[0]
    seg_len_list = np.insert(np.diff(seg_len_list), 0, first_seg_len)
    total_length = 0
    for i in range(len(state_list)):
        total_length += seg_len_list[i]
        if i == len(state_list)-1:
            new_seg_json[total_length] = state_list[i]
            break
        if state_list[i] == state_list[i+1]:
            continue
        else:
            new_seg_json[total_length] = state_list[i]
    return new_seg_json

def generate(s):
    channel_list = []
    # generate irrelevant channels.
    for i in range(num_relevant_channels):
        # seg_json = generate_seg_json(seg_len, num_states, length)
        seg_json = generate_seg_json([1200, 2500], [2,8], length)
        seg_json = merge_repeated_states(seg_json)
        result = gen_channel_from_json(seg_json).round(4)
        # print(result.shape)
        channel_list.append(result)
    # generate noise channels.
    for i in range(num_relevant_channels):
        result = np.random.normal(0, 1, length).round(4)
        channel_list.append(result)    
    # generate destructive channels.
    seg_json = {2000:0,4000:1,6000:2,8000:3,10000:4,12000:5,14000:6,16000:7,20000:8}
    for i in range(num_relevant_channels-2):
        result = gen_channel_from_json(seg_json, forking_depth=5).round(4)
        channel_list.append(result)
    temp_json_list = [
        {2000:0,4000:1,20000:2},
        {4000:0,6000:1,8000:2,20000:0},
        {8000:0,10000:1,12000:2,20000:0},
        {12000:0,14000:1,16000:2,20000:0}
    ]
    for i in range(num_relevant_channels):
        result = gen_channel_from_json(temp_json_list[i], forking_depth=5).round(4)
        channel_list.append(result)
    for i in range(num_relevant_channels):      
        result = gen_channel_from_json(temp_json_list[i], forking_depth=1).round(4)
        channel_list.append(result)
    seg_json = {2000:s+0,4000:s+1,6000:s+2,8000:s+3,10000:s+4,12000:s+5,14000:s+6,16000:s+7,20000:s+8}
    state_seq = seg_to_label(seg_json).astype(int)
    channel_list.append(state_seq)
    result = np.stack(channel_list).T
    return result

for i in range(6):
    result = generate(i*9)
    print(result.shape)
    np.save(f'{save_path}/synthetic{i+1}.npy', result)