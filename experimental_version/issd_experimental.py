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
import sys
sys.path.append('model')
from ts2vec import TS2Vec

class ISSD:
    def __init__(self,
        clustering_threshold=0.2,
        num_samples=30,
        min_seg_len_to_exclude=100,
        test_method='nn',
        inte_strategy='lda',
        n_jobs=10) -> None:

        self.clustering_threshold = clustering_threshold
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
            # print(data.shape)
            data = data.T
            # add a dimension to the last axis to make it 3D
            data_blc = np.expand_dims(data, axis=-1)
            # print(data.shape)
            rep_model = TS2Vec(1, 320, batch_size=4, max_train_length=1000, device='cuda:0')
            rep_model.fit(data_blc, n_iters=10)
            print('done')
            # cps = find_cut_points_from_state_seq(state_seq)
            _, cps = calculate_true_matrix_cf(state_seq)
            # cut into segments by cut points
            segments = [np.expand_dims(data[:,cps[i]:cps[i+1]], axis=-1) for i in range(len(cps)-1)]
            print(len(segments), len(cps))
            print(segments[0].shape)
            emb_list = []
            for seg in segments:
                emb = rep_model.encode(seg, encoding_window='full_series')
                print(emb.shape)
                emb_list.append(emb)
            embs = np.swapaxes(np.stack(emb_list), 0, 1)
            matrices_, true_matrix_, corr_matrix_ = compute_matrices(embs, state_seq, data.T)
            # print(matrices_.shape, true_matrix_.shape, corr_matrix_.shape)
            # matrices_: (num_channels, num_segs, num_segs)
            # true_matrix_: (num_segs, num_segs)
            # corr_matrix_: (num_channels, num_channels)

            self.matrices.append(matrices_.reshape(matrices_.shape[0], -1)) # flatten 1,2 dim
            self.true_matrices.append(true_matrix_.flatten())
            self.corr_matrices.append(corr_matrix_)
        self.corr_matrices = np.array(self.corr_matrices).mean(axis=0)
        self.clusters = cluster_corr(self.corr_matrices, threshold=self.clustering_threshold)
    
    def get_completeness_quality(self):
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
            masked_idx[self.clusters==self.clusters[idx]] = True
            current_matrix = matrix_OR(indicator_matrices[selected_channels_cf])
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
        self.inte_qf_cf()

    def inte_solution(self):
        score_qf = 0
        score_cf = 0
        for data, state_seq in zip(self.datalist, self.state_seq_list):
            reduced_data_issd_qf = data[:,self.qf_solution]
            reduced_data_issd_cf = data[:,self.qf_solution]
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

def compute_matrices(embs, state_seq, indicators):
    # GET BASIC INFORMATION
    num_channels = embs.shape[0]
    num_segs = embs.shape[1]
    # calculate true matrix
    true_matrix, cut_points = calculate_true_matrix_cf(state_seq)
    # CLUSTER INDICATORS BY PEARSON CORRELATION
    corr_matrix = pd.DataFrame(indicators).corr(method='pearson').to_numpy()
    corr_matrix[np.isnan(corr_matrix)] = 1 # two constant channels will yild nan
    # CALCULATE MATRICES
    matrix_list = []
    for i in range(num_channels):
        matrix = np.eye(num_segs)
        for j in range(num_segs):
            for k in range(num_segs):
                if j >= k:
                    continue
                else:
                    # calculate the distance between two embs
                    distance = np.linalg.norm(embs[i,j]-embs[i,k], ord=2)
                    matrix[j,k] = distance
                    matrix[k,j] = distance

        matrix_list.append(matrix)
    matrices = np.stack(matrix_list)
    print(matrices.shape, true_matrix.shape, corr_matrix.shape)
    return matrices, true_matrix, corr_matrix