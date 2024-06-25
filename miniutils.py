"""
Utils
"""

import numpy as np
from scipy import stats
import scipy.cluster.hierarchy as sch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.neighbors import KDTree, BallTree
import numpy as np
from scipy.signal import find_peaks

def is_constant(signal, tolerance=1e-5):
    return np.all(np.abs(np.diff(signal)) < tolerance)

def moving_average(signal, window_size=3):
    num_channels = signal.shape[1]
    channels = []
    for i in range(num_channels):
        if is_constant(signal[:, i]):
            # skip constant channel
            channels.append(signal[:, i])
            continue
        filtered = np.convolve(signal[:, i], np.ones(window_size) / window_size, mode='same')
        channels.append(filtered)
    signal = np.column_stack(channels)
    return signal

def reorder_label(label):
    # Start from 0.
    label = np.array(label)
    ordered_label_set = np.unique(compact(label)).astype(int)
    idx_list = [np.argwhere(label==e) for e in ordered_label_set]
    for i, idx in enumerate(idx_list):
        label[idx] = i
    return label.astype(int)

def majority_vote_on_results(result_list, K):
    results = np.array(result_list).flatten()
    elems, cnt = np.unique(results, return_counts=True)
    result = elems[np.argsort(cnt)[::-1][:K]]
    return result

def inte_issd(dataset, K, fname_list, strategy, clustering_threshold=0.2):
    fname_list = [fname[:-4] for fname in fname_list]
    if strategy == 'qf':
        # original version
        # interval_list = []
        # for fname in fname_list:
        #     mean_inner = np.load(f'output/issd-qf/{dataset}/{fname}/mean_inner.npy')
        #     mean_inter = np.load(f'output/issd-qf/{dataset}/{fname}/mean_inter.npy')
        #     interval_list.append(mean_inter-mean_inner)
        # interval_list = np.array(interval_list)
        # interval_list = np.mean(interval_list, axis=0)
        # return list(np.argsort(-interval_list)[:K])
        # new version
        matrices = []
        true_matrices = []
        corr_matrices = []
        for fname in fname_list:
            channel_matrices = np.load(f'output/issd-cf/{dataset}/{fname}/matrices.npy') # (num_channels, num_segs, num_segs)
            true_matrix = np.load(f'output/issd-cf/{dataset}/{fname}/true_matrix.npy') # (num_segs, num_segs)
            corr_matrix = np.load(f'output/issd-cf/{dataset}/{fname}/corr_matrix.npy') # (num_channels, num_channels)
            channel_matrices = channel_matrices.reshape(channel_matrices.shape[0], -1) # flatten 1,2 dim
            matrices.append(channel_matrices)
            corr_matrices.append(corr_matrix)
            true_matrices.append(true_matrix.flatten())
        if len(fname_list) == 1:
            matrices = matrices[0]
            true_matrices = true_matrices[0]
        else:
            matrices = np.hstack(matrices)
            true_matrices = np.hstack(true_matrices)
        corr_matrices = np.array(corr_matrices).mean(axis=0)
        clusters = cluster_corr(corr_matrices, threshold=clustering_threshold)
        # print(matrices.shape, true_matrices.shape, corr_matrices.shape)
        idx_inner = np.argwhere(true_matrices==True)
        idx_inter = np.argwhere(true_matrices==False)
        mean_inner = np.mean(matrices[:,idx_inner], axis=1).flatten()
        mean_inter = np.mean(matrices[:,idx_inter], axis=1).flatten()
        interval = mean_inter - mean_inner

        selected_channels = []
        masked_idx = np.array([False]*len(mean_inner))
        while len(selected_channels) < K:
            remaining_idx = np.argwhere(~masked_idx).flatten()
            if len(remaining_idx) == 0:
                print('No more channels to select.')
                break
            # select the idx with the maximum mean interval
            idx = remaining_idx[np.argmax(interval[remaining_idx])]
            selected_channels.append(idx)
            # mask the idx and the idx with the same cluster
            masked_idx[idx] = True
            masked_idx[clusters==clusters[idx]] = True
        return selected_channels
    elif strategy == 'cf':
        matrices = []
        true_matrices = []
        for fname in fname_list:
            channel_matrices = np.load(f'output/issd-cf/{dataset}/{fname}/matrices.npy') # (num_channels, num_segs, num_segs)
            true_matrix = np.load(f'output/issd-cf/{dataset}/{fname}/true_matrix.npy') # (num_segs, num_segs)
            channel_matrices = channel_matrices.reshape(channel_matrices.shape[0], -1) # flatten 1,2 dim
            matrices.append(channel_matrices)
            true_matrices.append(true_matrix.flatten())
        if len(fname_list) == 1:
            matrices = matrices[0]
            true_matrices = true_matrices[0]
        else:
            matrices = np.hstack(matrices)
            true_matrices = np.hstack(true_matrices)
        print(matrices.shape, true_matrices.shape)
        idx_inner = np.argwhere(true_matrices==True)
        idx_inter = np.argwhere(true_matrices==False)
        mean_inner = np.mean(matrices[:,idx_inner], axis=1).flatten()
        mean_inter = np.mean(matrices[:,idx_inter], axis=1).flatten()
        interval = mean_inter - mean_inner

        selected_channels = []
        masked_idx = np.array([False]*len(mean_inner))

        # STATIC FORWARD SELECTION
        completeness, indicator_matrices = cal_completeness(matrices, true_matrices)
        current_matrix = np.zeros(true_matrices.shape).astype(bool)
        while len(selected_channels) < K:
            remaining_idx = np.argwhere(~masked_idx).flatten()
            if len(remaining_idx) == 0:
                print('No more channels to select.')
                break
            cost_list = [cost(indicator_matrices[j], current_matrix) for j in range(indicator_matrices.shape[0])]
            cost_list = np.array(cost_list)
            candidate_c = np.max(cost_list[remaining_idx])
            candidate_idx = remaining_idx[np.argwhere(cost_list[remaining_idx]==candidate_c).flatten()]
            idx = candidate_idx[np.argmax(interval[candidate_idx])]
            selected_channels.append(idx)
            masked_idx[idx] = True
            current_matrix = matrix_OR([current_matrix, indicator_matrices[idx]])
        return selected_channels

        # STATIC BACKWARD SELECTION
        # completeness, indicator_matrices = cal_completeness(matrices, true_matrices)
        # # initialize with all channels
        # selected_channels = list(range(matrices.shape[0]))
        # while len(selected_channels) > K:
        #     remaining_idx = np.argwhere(~masked_idx).flatten()
        #     current_matrix = matrix_OR(indicator_matrices[selected_channels])
        #     cost_list = [cost(indicator_matrices[j], current_matrix) for j in range(indicator_matrices.shape[0])]
        #     cost_list = np.array(cost_list)
        #     candidate_c = np.min(cost_list[remaining_idx])
        #     candidate_idx = remaining_idx[np.argwhere(cost_list[remaining_idx]==candidate_c).flatten()]
        #     idx = candidate_idx[np.argmin(interval[candidate_idx])]
        #     selected_channels.remove(idx)
        #     masked_idx[idx] = True
        #     current_matrix = matrix_OR([current_matrix, indicator_matrices[idx]])
        # return selected_channels

        # DYNAMIC FORWARD SELECTION
        # while len(selected_channels) < K:
        #     remaining_idx = np.argwhere(~masked_idx).flatten()
        #     if len(remaining_idx) == 0:
        #         print('No more channels to select.')
        #         break
        #     # average the matrices of the selected channels
        #     if len(selected_channels) == 0: # empty
        #         current_matrix = np.zeros(true_matrices.shape).astype(bool)
        #     else:
        #         current_matrix = matrices[selected_channels].sum(axis=0)
        #     remaining_matrices = matrices[remaining_idx]
        #     # add current_matrix to each row of remaining_matrices
        #     # current_matrix is of shape (num_segs,)
        #     # remaining_matrices is of shape (num_remaining, num_segs)
        #     # result is of shape (num_remaining, num_segs)
        #     result = remaining_matrices + current_matrix
        #     completeness, _ = cal_completeness(result, true_matrices)
        #     # idx = remaining_idx[np.argmax(completeness)]
        #     candidate_c = np.min(completeness)
        #     candidate_idx = remaining_idx[np.argwhere(completeness==candidate_c).flatten()]
        #     idx = candidate_idx[np.argmax(interval[candidate_idx])]
        #     selected_channels.append(idx)
        #     masked_idx[idx] = True
        # return selected_channels

        # DYNAMIC BACKWARD SELECTION
        # initialize with all channels
        # selected_channels = list(range(matrices.shape[0]))
        # while len(selected_channels) > K:
        #     remaining_idx = np.argwhere(~masked_idx).flatten()
        #     current_matrix = matrices[selected_channels].sum(axis=0)
        #     # print(current_matrix.shape)
        #     remaining_matrices = matrices[remaining_idx]
        #     remaining_matrices = current_matrix - remaining_matrices
        #     # print(remaining_matrices.shape)
        #     completeness, _ = cal_completeness(remaining_matrices, true_matrices)
        #     candidate_c_for_removal = np.max(completeness)
        #     candidate_idx_for_removal = remaining_idx[np.argwhere(completeness==candidate_c_for_removal).flatten()]
        #     idx_for_removal = candidate_idx_for_removal[np.argmin(interval[candidate_idx_for_removal])]
        #     selected_channels.remove(idx_for_removal)
        #     masked_idx[idx_for_removal] = True
        # return selected_channels

# def cal_completeness(matrices, true_matrices):
#     idx_inner = np.argwhere(true_matrices==True)
#     idx_inter = np.argwhere(true_matrices==False)
#     num_channels = matrices.shape[0]
#     completeness = []
#     indicator_matrices = np.zeros_like(matrices).astype(bool)
#     for i in range(num_channels):
#         inner_tau = np.max(matrices[i, idx_inner].flatten())
#         c = np.sum(matrices[i]>inner_tau)
#         completeness.append(c)
#     return np.array(completeness), indicator_matrices

def cal_completeness(matrices, true_matrices):
    idx_inner = np.argwhere(true_matrices==True)
    idx_inter = np.argwhere(true_matrices==False)
    num_channels = matrices.shape[0]
    completeness = []
    indicator_matrices = np.zeros_like(matrices).astype(bool)
    for i in range(num_channels):
        inner_tau = matrices[i, idx_inner].flatten()
        inter_tau = matrices[i, idx_inter].flatten()
        inner_tau = np.sort(inner_tau)[::-1]
        inter_tau = np.sort(inter_tau)
        # find from where inner_tau is smaller than inter_tau
        for j in range(len(inner_tau)):
            if inner_tau[j] < inter_tau[j]:
                break
        completeness.append(j)
        idx_true1 = np.argwhere(matrices[i] > inter_tau[j]).flatten().tolist()
        # print(idx_true1)
        indicator_matrices[i, idx_true1] = True
        # union
        idx_true1 = list(set(idx_true1+idx_inter.flatten().tolist()))
        idx_true2 = np.argwhere(matrices[i] < inner_tau[j]).flatten().tolist()
        # union
        idx_true2 = list(set(idx_true2+idx_inner.flatten().tolist()))
        indicator_matrices[i, idx_true2] = True
    return np.array(completeness), indicator_matrices

def load_data(path):
    data = np.load(path, allow_pickle=True)
    state_seq = data[:,-1].astype(int)
    data = data[:,:-1]
    return data, state_seq

def adapt_for_clf(data, state_seq):
    cps = find_cut_points_from_state_seq(state_seq)
    segments = [data[cps[i]:cps[i+1]] for i in range(len(cps)-1)]
    # padding to the same length
    max_len = max([len(segment) for segment in segments])
    segments = [np.pad(segment, ((0,max_len-len(segment)),(0,0)), 'constant') for segment in segments]
    # cut to the same length
    # min_len = min([len(segment) for segment in segments])
    # segments = [segment[:min_len] for segment in segments]
    new_segments = []
    for seg in segments:
        new_seg = [pd.Series(e) for e in seg.T]
        new_segments.append(pd.Series(new_seg))
    segments = pd.DataFrame(new_segments)
    label = compact(state_seq)
    return segments, label

def matrix_OR(matrices):
    """
    Calculate the element-wise OR of a list of matrices.
    Equivalent to the \sum\vee operation in the paper.
    """
    if len(matrices) == 0:
        return None
    result = matrices[0]
    for i in range(1, len(matrices)):
        result = np.logical_or(result, matrices[i])
    return result

def is_complete(matrix, true_matrix):
    """
    judge if a channel is complete.
    """
    if np.array_equal(matrix, true_matrix):
        return True
    else:
        return False

def matrix_dist(mat1, mat2):
    """
    Calculate the distance between two matrices.
    """
    return np.sum(np.abs(mat1.astype(int)-mat2.astype(int)))

def count_true(matrix):
    """
    Count the number of True in a matrix.
    """
    return np.sum(matrix)

def count_false(matrix):
    """
    Count the number of False in a matrix.
    the numpy matrix may be 1-dim or 2-dim.
    """
    return matrix.size - np.sum(matrix)

def cost(mat, current_matrix):
    """
    Count where mat is True and current_matrix is False.
    """
    return np.sum(np.logical_and(np.logical_not(current_matrix), mat))

def plot_mts(X, groundtruth=None, prediction=None, figsize=(18,2), show=False):
    '''
    X: Time series, whose shape is (T, C) or (T, 1), (T, ) for uts, where T is length, C
        is the number of channels.
    groundtruth: can be of shape (T,) or (T, 1).
    prediction: can be of shape (T,) or (T, 1).
    '''
    if groundtruth is None and prediction is None:
        plt.plot(X)

    elif groundtruth is not None and prediction is not None:
        plt.figure(figsize=(16,4))
        # plt.style.use('classic')

        grid = plt.GridSpec(5,1)
        ax1 = plt.subplot(grid[0:3])
        plt.title('Time Series')
        plt.yticks([])
        plt.plot(X)

        # plt.style.use('classic')
        plt.subplot(grid[3], sharex=ax1)
        plt.title('State Sequence (Groundtruth)')
        plt.yticks([])
        plt.imshow(groundtruth.reshape(1, -1), aspect='auto', cmap='tab20c',
          interpolation='nearest')

        # plt.style.use('classic')
        plt.subplot(grid[4], sharex=ax1)
        plt.title('State Sequence (Prediction)')
        plt.yticks([])
        plt.imshow(prediction.reshape(1, -1), aspect='auto', cmap='tab20c',
          interpolation='nearest')

    else:
        if groundtruth is not None:
            plt.figure(figsize=(16,4))
            # plt.style.use('classic')

            grid = plt.GridSpec(4,1)
            ax1 = plt.subplot(grid[0:3])
            plt.title('Time Series')
            plt.yticks([])
            plt.plot(X)

            # plt.style.use('classic')
            plt.subplot(grid[3], sharex=ax1)
            plt.title('State Sequence')
            plt.yticks([])
            plt.imshow(groundtruth.reshape(1, -1), aspect='auto', cmap='tab20c',
            interpolation='nearest')

    plt.tight_layout()
    if show:
        plt.show()

def compact(series):
    '''
    Compact Time Series.
    '''
    compacted = []
    pre = series[0]
    compacted.append(pre)
    for e in series[1:]:
        if e != pre:
            pre = e
            compacted.append(e)
    return compacted

def state_seq_to_seg_json(state_seq):
    """
    Convert state sequence to json format label.
    The json format label is like: {100: 1, 200: 2, 300: 1, 400: 0},
    where the key is the index of change point, and the value is the state.
    The last change point is the end of the time series.
    """
    seg_json = {}
    pre = state_seq[0]
    for idx, e in enumerate(state_seq[1:]):
        if e != pre:
            seg_json[idx+1] = pre
            pre = e
    seg_json[len(state_seq)] = pre
    return seg_json

def calculate_acf(x, lags):
    """
    Calculate autocorrelation
    @Params:
        x: time series
        lags: max lag to calculate
    """
    x = x.copy()
    n = len(x)
    x = x.astype(float)
    result = [np.correlate(x[i:], x[:n-i]) for i in range(1, lags+1)]
    return np.array(result)

def find_k_by_acf(x, max_lag=500, default=50):
    """
    Find the subseries length k by acf.
    @Params:
        x: time series
    """
    x = x.copy()
    number_channels = x.shape[1]
    acf_list = []
    for channel_idx in range(number_channels):
        channel = x[:,channel_idx]
        # Calculate autocorrelation
        # use cross correlation
        # acf = np.correlate(channel, channel, mode='full')
        acf = calculate_acf(channel, max_lag)
        peak, _ = find_peaks(acf.flatten(), height=0, prominence=0.1)
        if len(peak)==0: # if no maxima, set to default.
            peak = default
        elif peak[0] > max_lag or peak[0] < 10: # if the maxima is out of range, set to default.
            peak = default
        else: # otherwise, set to the first peak.
            peak = peak[0]
        acf_list.append(peak)
    return np.array(acf_list)

def search_thresholds(matrices, true_matrix):
    """
    Search for the threshold to dertermine complete channel.
    """
    matrices = matrices.copy()
    num_channels = matrices.shape[0]
    inner_tau_list = []
    inter_tau_list = []
    inner_mean_tau_list = []
    inter_mean_tau_list = []
    inner_std_list = []
    inter_std_list = []
    idx_inner = np.argwhere(true_matrix==True)
    idx_inter = np.argwhere(true_matrix==False)
    # exclude the diagonal
    # idx_inner = idx_inner[idx_inner[:,0]!=idx_inner[:,1]]
    # idx_inter = idx_inter[idx_inter[:,0]!=idx_inter[:,1]]
    for i in range(num_channels):
        # get elements by indices
        inner_tau_list.append(np.max(matrices[i][idx_inner[:,0], idx_inner[:,1]]))
        inter_tau_list.append(np.min(matrices[i][idx_inter[:,0], idx_inter[:,1]]))
        inner_mean_tau_list.append(np.mean(matrices[i][idx_inner[:,0], idx_inner[:,1]]))
        inter_mean_tau_list.append(np.mean(matrices[i][idx_inter[:,0], idx_inter[:,1]]))
        inner_std_list.append(np.std(matrices[i][idx_inner[:,0], idx_inner[:,1]]))
        inter_std_list.append(np.std(matrices[i][idx_inter[:,0], idx_inter[:,1]]))
    return np.array(inner_tau_list),\
           np.array(inter_tau_list),\
           np.array(inner_mean_tau_list),\
           np.array(inter_mean_tau_list),\
           np.array(inner_std_list),\
           np.array(inter_std_list)

def compact_matrix(state_seq):
    elems, counts = np.unique(state_seq, return_counts=True)
    num_states = len(elems)
    compact_true_matrix = np.eye(num_states)
    compact_true_matrix = compact_true_matrix==1
    return compact_true_matrix

def cluster_corr(corr_array, threshold=0.2, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = corr_array
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_array = sch.fcluster(linkage, threshold, criterion='distance')
    return cluster_array

def pair_wise_nntest(segments, n, k, method='nn'):
    """
    channel should be of shape (num_samples,)
    """
    num_segments = len(segments)
    matrix = np.zeros((num_segments, num_segments))
    for i in range(num_segments):
         for j in range(num_segments):
            if i >= j:
                continue
            if method == 'ks':
                # result = stats.ks_2samp(segments[i], segments[j], method='asymp')
                result = stats.ks_2samp(segments[i], segments[j])
                matrix[i,j] = result.pvalue
                matrix[j,i] = result.pvalue
            elif method == 'nn':
                result = nn_test(segments[i], segments[j], n, k)
                matrix[i,j] = result
                matrix[j,i] = result
    return matrix

def pair_wise_nntest_wrapper(args):
    return pair_wise_nntest(**args)

def find_cut_points_from_state_seq(state_seq):
    # the last element in cut_point_list is the length of state_seq.
    cut_point_list = []
    c = state_seq[0]
    for i, e in enumerate(state_seq):
        if e == c:
            pass
        else:
            cut_point_list.append(i)
            c = e
    cut_point_list.insert(0, 0)
    cut_point_list.append(i+1)
    return cut_point_list

def calculate_true_matrix_from_state_seq(state_seq):
    """
    state_seq should be of shape (num_samples,)
    """
    compacted_seq = compact(state_seq)
    num_segments = len(compacted_seq)
    matrix = np.eye(num_segments)
    matrix = matrix==1
    for i in range(num_segments):
        for j in range(num_segments):
            if i >= j:
                continue
            if compacted_seq[i] == compacted_seq[j]:
                matrix[i,j] = True
                matrix[j,i] = True
    return matrix

def calculate_true_matrix(state_seq):
    """
    state_seq should be of shape (num_samples,)
    This function is for boundary-first version.
    """
    cut_points = find_cut_points_from_state_seq(state_seq)
    true_matrix = calculate_true_matrix_from_state_seq(state_seq)
    return true_matrix, cut_points

def calculate_true_matrix_cf(state_seq):
    """
    state_seq should be of shape (num_samples,)
    This function is for completeness-first version.
    When a state appears only once, split it into two segments.
    """
    cut_points = find_cut_points_from_state_seq(state_seq)
    compacted_seq = compact(state_seq)
    state, count = np.unique(compacted_seq, return_counts=True)
    state_list = []
    # split states into two segs when it only appears once.
    for s in compacted_seq:
        if count[state==s] == 1:
            state_list.append(s)
            state_list.append(s)
        else:
            state_list.append(s)
    # corresponding to the cut points
    cps = [0]
    cnt = 0
    for cp in cut_points[1:]:
        if count[state==compacted_seq[cnt]] == 1:
            cps.append(int((cp+cut_points[cnt])/2))
            cps.append(cp)
        else:
            cps.append(cp)
        cnt += 1
    num_segments = len(state_list)
    matrix = np.eye(num_segments)
    matrix = matrix==1
    for i in range(num_segments):
        for j in range(num_segments):
            if i >= j:
                continue
            if state_list[i] == state_list[j]:
                matrix[i,j] = True
                matrix[j,i] = True
    return matrix, cps

def exclude_trival_segments(state_seq, exclude_lenth):
    cps = find_cut_points_from_state_seq(state_seq)
    len_list = np.diff(cps)
    # idx_trival_segs = np.argwhere(len_list <= exclude_lenth)
    segs = []
    for seg_len in len_list:
        if seg_len > exclude_lenth:
            segs.append(np.ones(seg_len,dtype=bool))
        else:
            segs.append(np.zeros(seg_len,dtype=bool))
    non_trival_idx = np.concatenate(segs)
    return non_trival_idx

def sample_subseries(X, n, k):
    """
    sample n random subseries from x.
    X: time series sample
    n: num of subseries
    k: length of subseries
    """
    length = X.shape[0]
    # start_points = np.random.randint(0, length-k, n)
    start_points = np.linspace(0, length-k, n, dtype=int)
    # start_points = np.unique(np.linspace(0, length-k, n, dtype=int))
    segments = [X[sp:sp+k] for sp in start_points]
    return segments
    
def nn_test(sample1, sample2, n, k, nnmethod='ball'):
    """
    sample1: time series sample
    sample2: time series sample
    n: num of subseries
    k: length of subseries
    nnmethod: nearest neighbor query method, 'ball' or 'kd'
    """
    A1 = sample_subseries(sample1, n, k)
    A2 = sample_subseries(sample2, n, k)
    A1 = [e.flatten() for e in A1]
    A2 = [e.flatten() for e in A2]
    p = n*2
    r = int(math.log(p))
    A = A1 + A2
    T_rp = 0
    # construct KDTree or BallTree
    if nnmethod == 'kd':
        tree = KDTree(A, metric='euclidean', leaf_size=10)
    else: # use BallTree by default
        tree = BallTree(A, metric='euclidean', leaf_size=10)
    for i in range(p):
        dist, idx = tree.query(A[i].reshape(1,-1), k=r+1)
        idx = idx[0]
        for j in range(r):
            if (i < n and idx[j+1] < n) or (i >= n and idx[j+1] >= n):
                T_rp += 1
            else:
                continue
    return T_rp/(p*r)