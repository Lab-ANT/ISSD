from sklearn.feature_selection import mutual_info_regression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression

import sys
sys.path.append('.')
from miniutils import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os

datasets = ['MoCap', 'ActRecTut', 'USC-HAD', 'PAMAP2', 'SynSeg']

for dataset in datasets:
    print(dataset)
    fname_list = os.listdir(f'data/{dataset}/raw/')
    os.makedirs(f'data/{dataset}/issd', exist_ok=True)
    for fn_test in fname_list:
        score_qf = 0
        score_cf = 0
        count = 0
        for fname in fname_list:
            if fn_test == fname:
                continue
            reduced_data_issd_qf, state_seq = load_data(f'data/{dataset}/issd-qf/{fname}')
            pca_issd_qf = LinearDiscriminantAnalysis().fit_transform(reduced_data_issd_qf, state_seq)
            score_qf+=mutual_info_regression(pca_issd_qf, state_seq)[0]
            # ar = LinearRegression().fit(reduced_data_issd_qf, state_seq)
            # score_qf += ar.score(reduced_data_issd_qf, state_seq)

            reduced_data_issd_cf, state_seq = load_data(f'data/{dataset}/issd-cf/{fname}')
            pca_issd_cf = LinearDiscriminantAnalysis().fit_transform(reduced_data_issd_cf, state_seq)
            score_cf+=mutual_info_regression(pca_issd_cf, state_seq)[0]
            # ar = LinearRegression().fit(reduced_data_issd_cf, state_seq)
            # score_cf += ar.score(reduced_data_issd_cf, state_seq)
            count += 1
        s_qf = score_qf/count
        s_cf = score_cf/count
        if s_qf >= s_cf:
            print('issd', s_qf, s_cf, fn_test)
            data = np.load(f'data/{dataset}/issd-qf/{fn_test}', allow_pickle=True)
            np.save(f'data/{dataset}/issd/{fn_test}', data)
        else:
            print('issd-cf', s_qf, s_cf, fn_test)
            data = np.load(f'data/{dataset}/issd-cf/{fn_test}', allow_pickle=True)
            np.save(f'data/{dataset}/issd/{fn_test}', data)
