"""
Created by Chengyu on 2024/1/9.
Indicator Selection for State Detection.
ISSD v1.0.
"""

import numpy as np
from miniutils import *
import multiprocessing
from sklearn.preprocessing import StandardScaler

def selection_strategy_cf(min_inter, max_inner, mean_inter, mean_inner, std_inner, std_inter, cluster, matrices, true_matrix, K):
    """
    An equivalent implementation of the algorithm in the paper.
    """
    true_matrix = true_matrix.copy()
    true_matrix = ~true_matrix
    indicator_matrices = np.array([m>tau for m, tau in zip(matrices, max_inner)])

    masked_idx = np.array([False]*len(min_inter))
    mean_interval = mean_inter-mean_inner-std_inner-std_inter
    
    selection_results = []
    # COMPLETENESS GUARANTEE
    # current_matrix = matrix_OR(indicator_matrices[selection_results])
    current_matrix = np.zeros(true_matrix.shape).astype(bool)
    # while not is_complete(current_matrix, true_matrix):
    while len(selection_results) < K:
        remaining_idx = np.argwhere(~masked_idx).flatten()
        if len(remaining_idx) == 0 or len(selection_results) >= K:
            break
        cost_list = np.array([cost(m, current_matrix) for m in indicator_matrices])
        if len(np.argwhere(cost_list>0)) == 0:
            break
        candidate_c = np.max(cost_list[remaining_idx])
        candidate_idx = remaining_idx[np.argwhere(cost_list[remaining_idx]==candidate_c).flatten()]
        idx = candidate_idx[np.argmax(mean_interval[candidate_idx])]
        selection_results.append(idx)
        # mask the idx and the idx with the same cluster
        masked_idx[idx] = True
        masked_idx[cluster==cluster[idx]] = True
        current_matrix = matrix_OR(indicator_matrices[selection_results])

    return selection_results, indicator_matrices

def selection_strategy_qf(min_inter, max_inner, mean_inter, mean_inner, std_inner, std_inter, cluster, matrices, true_matrix, K):
    """
    An equivalent implementation of the algorithm in the paper.
    """
    indicator_matrices = np.array([m>tau for m, tau in zip(matrices, max_inner)])

    masked_idx = np.array([False]*len(min_inter))
    selection_results = []
    mean_interval = mean_inter-mean_inner-std_inner-std_inter

    while len(selection_results) < K:
        remaining_idx = np.argwhere(~masked_idx).flatten()
        if len(remaining_idx) == 0:
            break
        # select the idx with the maximum mean interval
        idx = remaining_idx[np.argmax(mean_interval[remaining_idx])]
        selection_results.append(idx)
        # mask the idx and the idx with the same cluster
        masked_idx[idx] = True
        masked_idx[cluster==cluster[idx]] = True

    return selection_results, indicator_matrices

def issd(indicators, state_seq, K,
         strategy='qf',
         cluster_threshold=0.2,
         num_samples=30,
         # subseries_length=50,
         min_seg_len_to_exclude=50,
         two_sample_method = 'nn',
         save_path=None,
         n_jobs=10,
         plot=None):
    """
    Indicator Selection for State Detection.

    Parameters
    ----------
    indicators : array-like of shape (n_samples, n_channels)  
    state_seq : array-like of shape (n_samples,)
    K : int, desired number of channels.
    strategy : str, [qf, cf], optional, strategy for selection, qf for boundary-first
        and cf for completeness-first. Default is qf.
    cluster_threshold : float, optional, threshold for clustering. Default is 0.8.
    num_samples : int, optional, number of samples for nntest. Default is 50.
    subseries_length : int, optional, subseries length for nntest. Default is 200.
    min_seg_len_to_exclude : int, optional, segments of length lower than this value
        will be excluded. Default is 10.
    two_sample_method : str, [nn, ks], optional, method for two-sample test, ks for Kolmogorov-Smirnov
        test and nn for nearst-neighbor test. Default is nn.
    save_path : str, optional, if not None, save the intermediate results to the path.
    n_jobs : int, optional, number of jobs for parallel computing. Default is 10.

    Returns
    -------
    idx_selected_channels : array-like of shape (n_selected_channels,)

    Notes
    -----
    - The indicators should has scalar values.
    - The state_seq should be a sequence of integers.
    """

    indicators = indicators.copy()
    win_size = 10
    offset = win_size // 2
    indicators = moving_average(indicators, window_size=win_size)
    indicators = indicators[offset:-offset,:]
    state_seq = state_seq[offset:-offset]
    # PARAMS CHECK
    if strategy not in ['cf', 'qf']:
        raise ValueError('strategy must be cf or qf.')

    # CHECK DATA VALIDITY
    # check if indicators and state_seq have the same length
    if len(indicators) != len(state_seq):
        raise ValueError('indicators and state_seq have different lengths.')
    # check if indicators and stete_seq are numpy arrays
    if not isinstance(indicators, np.ndarray):
        raise TypeError('indicators should be a numpy array of shape (n_samples, n_channels).')
    if not isinstance(state_seq, np.ndarray):
        raise TypeError('state_seq should be a numpy array of shape (n_samples).')
    
    # IMPORT LIBRARIES AS NEEDED
    if save_path is not None:
        import matplotlib.pyplot as plt
        import os
        os.makedirs(save_path, exist_ok=True)

    # # EXCLUDE TRIVAL SEGMENTS
    non_trival_idx = exclude_trival_segments(state_seq, min_seg_len_to_exclude)
    indicators = indicators[non_trival_idx]
    state_seq = state_seq[non_trival_idx]

    # GET BASIC INFORMATION
    num_channels = indicators.shape[1]
    # calculate true matrix
    method_ctm = calculate_true_matrix if strategy == 'qf' else calculate_true_matrix_cf
    true_matrix, cut_points = method_ctm(state_seq)
    min_seg_len = np.min(np.diff(cut_points))
    acf = find_k_by_acf(indicators,
                        min_seg_len if min_seg_len<500 else 500,
                        default=min_seg_len-1)

    # CLUSTER INDICATORS BY PEARSON CORRELATION
    corr_matrix = pd.DataFrame(indicators).corr(method='pearson').to_numpy()
    corr_matrix[np.isnan(corr_matrix)] = 1 # two constant channels will yild nan
    cluster = cluster_corr(corr_matrix, threshold=cluster_threshold)

    # CALACULATE OR LOAD MATRICES
    if save_path is not None and os.path.exists(os.path.join(save_path, 'matrices.npy')):
        print('Existing results found, load saved matrices.')
        matrices = np.load(os.path.join(save_path, 'matrices.npy'))
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
        pool_args = []
        for channel_id in range(num_channels):
            segments = [indicators[cut_points[i]:cut_points[i+1],channel_id] for i in range(len(cut_points)-1)]
            arg = {'segments' : segments,
                   'n' : num_samples,
                   'k' : acf[channel_id],
                   'method' : two_sample_method}
            pool_args.append(arg)
        res = pool.map_async(pair_wise_nntest_wrapper, pool_args)
        pool.close()
        pool.join()
        matrices = res.get()
        matrices = np.stack(matrices)

    # SEARCH THRESHOLD
    max_inner, min_inter, mean_inner, mean_inter, std_inner, std_inter = search_thresholds(matrices, true_matrix)
    # compact_true_matrix = compact_matrix(state_seq)
    if strategy == 'qf':
        result, indicator_matrices = selection_strategy_qf(min_inter, max_inner,
                                                           mean_inter, mean_inner,
                                                           std_inner, std_inter,
                                                           cluster, matrices,
                                                           true_matrix, K)
    else: # cf
        result, indicator_matrices = selection_strategy_cf(min_inter, max_inner,
                                                           mean_inter, mean_inner,
                                                           std_inner, std_inter,
                                                           cluster, matrices,
                                                           true_matrix, K)

    if save_path is not None:
        # save matrices
        np.save(os.path.join(save_path, 'matrices.npy'), matrices)
        np.save(os.path.join(save_path, 'true_matrix.npy'), true_matrix)
        # save intervals
        np.save(os.path.join(save_path, 'min_inter.npy'), min_inter)
        np.save(os.path.join(save_path, 'max_inner.npy'), max_inner)
        np.save(os.path.join(save_path, 'mean_inter.npy'), mean_inter)
        np.save(os.path.join(save_path, 'mean_inner.npy'), mean_inner)
        np.save(os.path.join(save_path, 'std_inter.npy'), std_inter)
        np.save(os.path.join(save_path, 'std_inner.npy'), std_inner)
        # save cluster
        np.save(os.path.join(save_path, 'cluster.npy'), cluster)
    if plot is not None:
        # plot indicators
        plot_mts(StandardScaler().fit_transform(indicators), state_seq)
        plt.savefig(os.path.join(save_path, 'mts.png'))
        plt.close()
        # plot selected channels
        plot_mts(StandardScaler().fit_transform(indicators[:,result]), state_seq)
        plt.savefig(os.path.join(save_path, 'selected_channels.png'))
        plt.close()
        fig, ax = plt.subplots(nrows=K, figsize=(10, K*2))
        for i,idx in enumerate(result):
            ax[i].plot(indicators[:,idx])
        plt.savefig(os.path.join(save_path, f'test.png'))
        plt.close()
        # plot indicator matrices
        width = int(math.sqrt(num_channels))+1
        fig, ax = plt.subplots(nrows=width, ncols=width, figsize=(20,20))
        for i in range(width):
            for j in range(width):
                ax[i,j].set_yticks([])
                ax[i,j].set_xticks([])
                if i*width+j >= num_channels:
                    continue
                # print(i*width+j in result)
                if i*width+j in result:
                    ax[i,j].imshow(matrices[i*width+j], cmap='GnBu_r')
                else:
                    ax[i,j].imshow(matrices[i*width+j], cmap='gray')
                # if i*width+j in [38, 26, 55, 48]: # human
                #     # set the border of the selected channels
                #     for spine in ax[i,j].spines.values():
                #         spine.set_edgecolor('green')
                #         spine.set_linewidth(10)
        plt.savefig(os.path.join(save_path, 'matrices.png'))
        plt.close()
        # plot indicator matrices
        width = int(math.sqrt(num_channels))+1
        fig, ax = plt.subplots(nrows=width, ncols=width, figsize=(20,20))
        for m in indicator_matrices:
            m.astype(int)
        for i in range(width):
            for j in range(width):
                if i*width+j >= num_channels:
                    break
                if i*width+j in result:
                    ax[i,j].imshow(indicator_matrices[i*width+j], cmap='GnBu_r')
                else:
                    ax[i,j].imshow(indicator_matrices[i*width+j], cmap='gray')
                # if i*width+j in [38, 26, 55, 48]: # human
                #     # set the border of the selected channels
                #     for spine in ax[i,j].spines.values():
                #         spine.set_edgecolor('green')
                #         spine.set_linewidth(10)
                # ax[i,j].imshow(indicator_matrices[i*width+j], cmap='gray')
        plt.savefig(os.path.join(save_path, 'indicator_matrices.png'))
        plt.close()
        # plot true matrix
        plt.imshow(true_matrix)
        plt.savefig(os.path.join(save_path, 'true_matrix.png'))
        plt.close()
    return result