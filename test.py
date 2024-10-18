import numpy as np
import time
from itertools import groupby

# 定义函数：itertools 版本
def compress_sequence(seq):
    return [key for key, _ in groupby(seq)]

# 定义函数：numpy 版本
def compress_sequence_np(arr):
    arr = np.asarray(arr)
    return arr[np.insert(arr[1:] != arr[:-1], 0, True)]

# 测试用例生成随机序列
def generate_random_sequence(size):
    return np.random.randint(0, 10, size).tolist()

# 运行时间测试
def measure_time(func, seq):
    start_time = time.time()
    func(seq)
    return time.time() - start_time

# 测试函数
def test_compression():
    sizes = [1000, 5000, 10000, 50000]  # 各种序列大小
    num_repeats = 1000
    results = []

    for size in sizes:
        np_time_total = 0
        itertools_time_total = 0
        
        for _ in range(num_repeats):
            # 随机生成测试序列
            sequence = generate_random_sequence(size)

            # 测试 numpy 版本
            np_time_total += measure_time(compress_sequence_np, sequence)

            # 测试 itertools 版本
            itertools_time_total += measure_time(compress_sequence, sequence)
        
        # 计算平均时间
        np_avg_time = np_time_total / num_repeats
        itertools_avg_time = itertools_time_total / num_repeats

        # 保存结果
        results.append({
            'size': size,
            'numpy_avg_time': np_avg_time,
            'itertools_avg_time': itertools_avg_time
        })
    
    return results

# 运行测试并保存结果
test_results = test_compression()
import pandas as pd

df = pd.DataFrame(test_results)
print(df)
# import ace_tools as tools; tools.display_dataframe_to_user(name="Compression Algorithm Performance Comparison", dataframe=df)
