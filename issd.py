"""
Indicator Selection for State Detection.
ISSD v1.0.
"""

import numpy as np
from miniutils import *
import multiprocessing
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import itertools

class ISSD:
    def __init__(self,
        corr_threshold=0.8,
        num_samples=30,
        min_seg_len_to_exclude=100,
        test_method='nn',
        inte_strategy='lda',
        n_jobs=10) -> None:

        self.clustering_threshold = 1-corr_threshold
        self.num_samples = num_samples
        self.min_seg_len_to_exclude = min_seg_len_to_exclude
        self.test_method = test_method
        self.inte_strategy = inte_strategy
        self.n_jobs = n_jobs

    def compute_matrices(self, datalist, state_seq_list):
        self.datalist = datalist
        self.state_seq_list = state_seq_list
        self.matrices = []
        self.true_matrices = []
        self.corr_matrices = []
        for data, state_seq in zip(datalist, state_seq_list):
            matrices_, true_matrix_, corr_matrix_ = compute_matrices(data,
                                                                state_seq, 
                                                                self.num_samples,
                                                                self.min_seg_len_to_exclude,
                                                                self.test_method,
                                                                self.n_jobs)
            # print(matrices_.shape, true_matrix_.shape, corr_matrix_.shape)
            # matrices_: (num_channels, num_segs, num_segs)
            # true_matrix_: (num_segs, num_segs)
            # corr_matrix_: (num_channels, num_channels)

            self.matrices.append(matrices_.reshape(matrices_.shape[0], -1)) # flatten 1,2 dim
            self.true_matrices.append(true_matrix_.flatten())
            self.corr_matrices.append(corr_matrix_)
        self.corr_matrices = np.array(self.corr_matrices).mean(axis=0)
        self.clusters = cluster_corr(self.corr_matrices, threshold=self.clustering_threshold)
    
    def get_qf_solution(self, K):
        # QF strategy
        if len(self.matrices) == 1:
            stacked_matrices = self.matrices[0]
            stacked_true_matrices = self.true_matrices[0]
        else:
            stacked_matrices = np.hstack(self.matrices)
            stacked_true_matrices = np.hstack(self.true_matrices)
        # print(matrices.shape, true_matrices.shape, corr_matrices.shape)
        idx_inner = np.argwhere(stacked_true_matrices==True)
        idx_inter = np.argwhere(stacked_true_matrices==False)
        mean_inner = np.mean(stacked_matrices[:,idx_inner], axis=1).flatten()
        mean_inter = np.mean(stacked_matrices[:,idx_inter], axis=1).flatten()
        interval = mean_inter - mean_inner

        selected_channels_qf = []
        masked_idx = np.array([False]*len(mean_inner))
        while len(selected_channels_qf) < K:
            remaining_idx = np.argwhere(~masked_idx).flatten()
            if len(remaining_idx) == 0:
                print('No more channels to select.')
                break
            # select the idx with the maximum mean interval
            idx = remaining_idx[np.argmax(interval[remaining_idx])]
            selected_channels_qf.append(idx)
            # mask the idx and the idx with the same cluster
            masked_idx[idx] = True
            masked_idx[self.clusters==self.clusters[idx]] = True
        self.qf_solution = selected_channels_qf
        return selected_channels_qf

    def get_cf_solution(self, K):
        # CF strategy
        ts_interval = []
        indicator_matrices = []
        for ts_matrices, ts_true_matrix in zip(self.matrices, self.true_matrices):
            idx_inner = np.argwhere(ts_true_matrix==True)
            idx_inter = np.argwhere(ts_true_matrix==False)
            mean_inner = np.mean(ts_matrices[:,idx_inner], axis=1).flatten()
            mean_inter = np.mean(ts_matrices[:,idx_inter], axis=1).flatten()
            max_inner = np.max(ts_matrices[:,idx_inner], axis=1).flatten()
            interval = mean_inter - mean_inner
            ts_interval.append(interval)
            indicator_matrices_ = [m>tau for m, tau in zip(ts_matrices, max_inner)]
            # the following two lines are experimental
            # remove the above line and uncomment the following two lines
            # min_inter = np.min(ts_matrices[:,idx_inter], axis=1).flatten()
            # indicator_matrices_ = [m<tau for m, tau in zip(ts_matrices, min_inter)]
            indicator_matrices_ = np.array(indicator_matrices_, dtype=bool)
            indicator_matrices.append(indicator_matrices_)
        interval = np.array(ts_interval).mean(axis=0)
        if len(indicator_matrices) == 1:
                indicator_matrices = indicator_matrices[0]
        else:
            indicator_matrices = np.hstack(indicator_matrices)

        selected_channels_cf = []
        masked_idx = np.array([False]*len(mean_inner))

        current_matrix = np.zeros(indicator_matrices.shape).astype(bool)
        self.indicator_matrices = []
        while len(selected_channels_cf) < K:
            remaining_idx = np.argwhere(~masked_idx).flatten()
            costlist = np.array([cost(m, current_matrix) for m in indicator_matrices])
            candidate_c = np.max(costlist[remaining_idx])
            candidate_idx = remaining_idx[np.argwhere(costlist[remaining_idx]==candidate_c).flatten()]
            idx = candidate_idx[np.argmax(interval[candidate_idx])]
            selected_channels_cf.append(idx)
            masked_idx[idx] = True
            masked_idx[self.clusters==self.clusters[idx]] = True
            current_matrix = matrix_OR(indicator_matrices[selected_channels_cf])
            # the following line is for case-study-completeness.py
            print(idx, np.sum(current_matrix))
            # not used in the normal process
            # mean_quality = np.sum(interval[selected_channels_cf])/len(selected_channels_cf)
            # mean_quality = np.sum(interval[selected_channels_cf])
            # print(idx, np.sum(current_matrix), mean_quality)
            # self.indicator_matrices.append(current_matrix)
        # self.indicator_matrices = indicator_matrices

        self.cf_solution = selected_channels_cf
        return selected_channels_cf
    
    def exhaustively_search_cf(self, K):
        ts_interval = []
        indicator_matrices = []
        for ts_matrices, ts_true_matrix in zip(self.matrices, self.true_matrices):
            idx_inner = np.argwhere(ts_true_matrix==True)
            idx_inter = np.argwhere(ts_true_matrix==False)
            mean_inner = np.mean(ts_matrices[:,idx_inner], axis=1).flatten()
            mean_inter = np.mean(ts_matrices[:,idx_inter], axis=1).flatten()
            max_inner = np.max(ts_matrices[:,idx_inner], axis=1).flatten()
            interval = mean_inter - mean_inner
            ts_interval.append(interval)
            indicator_matrices_ = [m>tau for m, tau in zip(ts_matrices, max_inner)]
            indicator_matrices_ = np.array(indicator_matrices_, dtype=bool)
            indicator_matrices.append(indicator_matrices_)
        interval = np.array(ts_interval).mean(axis=0)
        if len(indicator_matrices) == 1:
                indicator_matrices = indicator_matrices[0]
        else:
            indicator_matrices = np.hstack(indicator_matrices)

        current_completeness = 0
        current_quality = 0
        combinations = list(itertools.combinations(range(indicator_matrices.shape[0]), K))
        solutions = iter(combinations)
        for selected_channels_cf in solutions:
            selected_channels_cf = list(selected_channels_cf)
            completeness = np.sum(indicator_matrices[selected_channels_cf])
            quality = np.sum(interval[selected_channels_cf])
            if completeness > current_completeness and quality > current_quality:
                current_completeness = completeness
                current_quality = quality
                self.true_cf_solution = selected_channels_cf
        return self.true_cf_solution

    def fit(self, datalist, state_seq_list, K):
        self.compute_matrices(datalist, state_seq_list)
        self.get_qf_solution(K)
        self.get_cf_solution(K)
        self.inte_solution()

    def compute_completeness_quality(self):
        """
        For analysis purpose, will not be used in the selection process.
        """
        # check if self.matrices exists
        if not hasattr(self, 'matrices'):
            raise ValueError('Please compute matrices by calling compute_matrices() first.')
        c_list = []
        q_list = []
        for mat, true_mat in zip(self.matrices, self.true_matrices):
            completeness = cal_completeness_experimental(mat, true_mat) # completeness is a np.array
            quality = cal_quality(mat, true_mat) # quality is a np.array
            c_list.append(completeness)
            q_list.append(quality)
        self.completeness = np.array(c_list)
        self.quality = np.array(q_list)
        # max_c = matrix_OR(self.)

    def inte_solution(self):
        score_qf = 0
        score_cf = 0
        for data, state_seq in zip(self.datalist, self.state_seq_list):
            reduced_data_issd_qf = data[:,self.qf_solution]
            reduced_data_issd_cf = data[:,self.cf_solution]
            if self.inte_strategy == 'lda':
                lda_issd_qf = LinearDiscriminantAnalysis(n_components=1).fit_transform(reduced_data_issd_qf, state_seq)
                score_qf += np.sum(mutual_info_regression(lda_issd_qf, state_seq))
                lda_issd_cf = LinearDiscriminantAnalysis(n_components=1).fit_transform(reduced_data_issd_cf, state_seq)
                score_cf += np.sum(mutual_info_regression(lda_issd_cf, state_seq))
            elif self.inte_strategy == 'pca':
                pca_issd_qf = PCA(n_components=1).fit_transform(reduced_data_issd_qf, state_seq)
                score_qf += np.sum(mutual_info_regression(pca_issd_qf, state_seq))
                pca_issd_cf = PCA(n_components=1).fit_transform(reduced_data_issd_cf, state_seq)
                score_cf += np.sum(mutual_info_regression(pca_issd_cf, state_seq))
            elif self.inte_strategy == 'mi':
                score_qf += np.sum(mutual_info_regression(reduced_data_issd_qf, state_seq))
                score_cf += np.sum(mutual_info_regression(reduced_data_issd_cf, state_seq))

        if score_qf >= score_cf:
            self.solution = self.qf_solution
        else:
            self.solution = self.cf_solution
        return self.solution
        
def issd(datalist, state_seq_list, K,
         clustering_threshold=0.2,
         num_samples=30,
         min_seg_len_to_exclude=100,
         test_method='nn',
         n_jobs=10):
    matrices = []
    true_matrices = []
    corr_matrices = []
    for data, state_seq in zip(datalist, state_seq_list):
        matrices_, true_matrix_, corr_matrix_ = compute_matrices(data,
                                                            state_seq, 
                                                            num_samples,
                                                            min_seg_len_to_exclude,
                                                            test_method,
                                                            n_jobs)
        # print(matrices_.shape, true_matrix_.shape, corr_matrix_.shape)
        # matrices_: (num_channels, num_segs, num_segs)
        # true_matrix_: (num_segs, num_segs)
        # corr_matrix_: (num_channels, num_channels)

        matrices.append(matrices_.reshape(matrices_.shape[0], -1)) # flatten 1,2 dim
        true_matrices.append(true_matrix_.flatten())
        corr_matrices.append(corr_matrix_)
    corr_matrices = np.array(corr_matrices).mean(axis=0)
    clusters = cluster_corr(corr_matrices, threshold=clustering_threshold)
    
    # QF strategy
    if len(matrices) == 1:
        stacked_matrices = matrices[0]
        stacked_true_matrices = true_matrices[0]
    else:
        stacked_matrices = np.hstack(matrices)
        stacked_true_matrices = np.hstack(true_matrices)
    # print(matrices.shape, true_matrices.shape, corr_matrices.shape)
    idx_inner = np.argwhere(stacked_true_matrices==True)
    idx_inter = np.argwhere(stacked_true_matrices==False)
    mean_inner = np.mean(stacked_matrices[:,idx_inner], axis=1).flatten()
    mean_inter = np.mean(stacked_matrices[:,idx_inter], axis=1).flatten()
    interval = mean_inter - mean_inner

    selected_channels_qf = []
    masked_idx = np.array([False]*len(mean_inner))
    while len(selected_channels_qf) < K:
        remaining_idx = np.argwhere(~masked_idx).flatten()
        if len(remaining_idx) == 0:
            print('No more channels to select.')
            break
        # select the idx with the maximum mean interval
        idx = remaining_idx[np.argmax(interval[remaining_idx])]
        selected_channels_qf.append(idx)
        # mask the idx and the idx with the same cluster
        masked_idx[idx] = True
        masked_idx[clusters==clusters[idx]] = True

    # CF strategy
    ts_interval = []
    indicator_matrices = []
    for ts_matrices, ts_true_matrix in zip(matrices, true_matrices):
        idx_inner = np.argwhere(ts_true_matrix==True)
        idx_inter = np.argwhere(ts_true_matrix==False)
        mean_inner = np.mean(ts_matrices[:,idx_inner], axis=1).flatten()
        mean_inter = np.mean(ts_matrices[:,idx_inter], axis=1).flatten()
        max_inner = np.max(ts_matrices[:,idx_inner], axis=1).flatten()
        interval = mean_inter - mean_inner
        ts_interval.append(interval)
        indicator_matrices_ = [m>tau for m, tau in zip(ts_matrices, max_inner)]
        indicator_matrices_ = np.array(indicator_matrices_, dtype=bool)
        indicator_matrices.append(indicator_matrices_)
    interval = np.array(ts_interval).mean(axis=0)
    if len(indicator_matrices) == 1:
            indicator_matrices = indicator_matrices[0]
    else:
        indicator_matrices = np.hstack(indicator_matrices)

    selected_channels_cf = []
    masked_idx = np.array([False]*len(mean_inner))

    current_matrix = np.zeros(indicator_matrices.shape).astype(bool)
    while len(selected_channels_cf) < K:
        remaining_idx = np.argwhere(~masked_idx).flatten()
        costlist = np.array([cost(m, current_matrix) for m in indicator_matrices])
        candidate_c = np.max(costlist[remaining_idx])
        candidate_idx = remaining_idx[np.argwhere(costlist[remaining_idx]==candidate_c).flatten()]
        idx = candidate_idx[np.argmax(interval[candidate_idx])]
        selected_channels_cf.append(idx)
        masked_idx[idx] = True
        masked_idx[clusters==clusters[idx]] = True
        current_matrix = matrix_OR(indicator_matrices[selected_channels_cf])

    return selected_channels_qf, selected_channels_cf

def compute_matrices(indicators, state_seq,
    num_samples=30,
    min_seg_len_to_exclude=100,
    two_sample_method='nn',
    n_jobs=10,
    save_path=None):
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
        os.makedirs(save_path, exist_ok=True)

    # COMPUTE OR LOAD MATRICES
    if save_path is not None and os.path.exists(os.path.join(save_path, 'matrices.npy')):
        print('Existing results found, load saved matrices.')
        matrices = np.load(os.path.join(save_path, 'matrices.npy'))
        true_matrix = np.load(os.path.join(save_path, 'true_matrix.npy'))
        corr_matrix = np.load(os.path.join(save_path, 'corr_matrix.npy'))
    else: # COMPUTE
        # EXCLUDE TRIVAL SEGMENTS
        non_trival_idx = exclude_trival_segments(state_seq, min_seg_len_to_exclude)
        indicators = indicators[non_trival_idx]
        state_seq = state_seq[non_trival_idx]
        # GET BASIC INFORMATION
        num_channels = indicators.shape[1]
        # calculate true matrix
        true_matrix, cut_points = calculate_true_matrix_cf(state_seq)
        # true_matrix, cut_points = calculate_true_matrix(state_seq)
        min_seg_len = np.min(np.diff(cut_points))
        acf = find_k_by_acf(indicators,
                            min_seg_len if min_seg_len<500 else 500,
                            default=min_seg_len-1)
        # CLUSTER INDICATORS BY PEARSON CORRELATION
        corr_matrix = pd.DataFrame(indicators).corr(method='pearson').to_numpy()
        corr_matrix[np.isnan(corr_matrix)] = 1 # two constant channels will yild nan
        # CALCULATE MATRICES
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

        if save_path is not None:
            # save matrices
            np.save(os.path.join(save_path, 'matrices.npy'), matrices)
            np.save(os.path.join(save_path, 'true_matrix.npy'), true_matrix)
            np.save(os.path.join(save_path, 'corr_matrix.npy'), corr_matrix)

    return matrices, true_matrix, corr_matrix