"""
Created by Chengyu on 2023/10/20.
Convert .amc files in MoCap dataset to .csv and .npy fromat for easy handling.
The first 3 lines of .amc files are comments and not used.
Every frame starts with a line of number indicating the number of the frame.
Refer to any .amc file in the dataset for the format of the frame.
"""

import os
import pandas as pd
import numpy as np
import tqdm

def parse_col_names(frame):
    """
    Parse the column names from any frame.
    """
    col_names = []
    raw_name_count = [(line.split(' ')[0], len(line.split(' '))-1) for line in frame]
    for name, count in raw_name_count:
        col_names+= [name if count==1 else name + str(i+1) for i in range(count)]
    return col_names


def parse_frame(frame):
    """
    Parse a frame into a numpy array.
    a frame is a list of string here.
    each string contains a joint's name and its following values,
    separated by a space.
    """
    values = np.concatenate([np.array(line.split(' ')[1:], dtype=float) for line in frame])
    return values

def parse_amc(path):
    """
    Parse a .amc file into a pandas dataframe.
    return: a pandas dataframe.
    """
    with open(path) as f:
        # load and remove the first 3 lines
        lines = f.readlines()[3:]
        # remove '\n' at the end of each line
        lines = [line.strip('\n') for line in lines]
        # cut into frames.
        # A frame consist of 30 lines, the first line is the number of the frame
        # the rest 29 lines are the content of the frame.
        frames = [lines[i+1:i+30] for i in range(0, len(lines), 30)]
        col_names = parse_col_names(frames[0])
        # parse each frame
        frames = [parse_frame(frame) for frame in frames]
        data = np.stack(frames).round(4)
        # assemble the dataframe
        df = pd.DataFrame(data, columns=col_names)
        return df

if __name__ == '__main__':
    script_path = os.path.dirname(__file__)
    # configure the relative path to the data folder.
    data_path = os.path.join(script_path, '../data/raw/MoCap-raw')
    # npy_save_path = os.path.join(script_path, 'MoCap_npy')
    csv_save_path = os.path.join(script_path, '../data/raw/MoCap-csv')
    # create path if not exist
    # os.makedirs(npy_save_path, exist_ok=True)
    os.makedirs(csv_save_path, exist_ok=True)
    fname_list = os.listdir(data_path)
    # filter non .amc files
    fname_list = [fname for fname in fname_list if fname.endswith('.amc')]
    for fname in tqdm.tqdm(fname_list):
        df = parse_amc(os.path.join(data_path, fname))
        df.to_csv(os.path.join(csv_save_path, fname[:-4]+'.csv'), index=False)