import sys
sys.path.append('.')
sys.path.append('downstream_methods/')
from miniutils import find_cut_points_from_state_seq
# GHMM
from hmmlearn.hmm import GaussianHMM
# TICC
from downstream_methods.TICC.TICC_solver import TICC
# ClaSP
from downstream_methods.ClaSP.clasp import ModifiedClaSP
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class DownstreamMethodAdaper():
    def __init__(self, method):
        if method not in ['time2state', 'ghmm', 'ticc', 'clasp', 'e2usd', 'constant']:
            raise ValueError('The downstream method does not exist.\
                             The supported methods are: time2state, ghmm, ticc, clasp, gmmhmm, e2usd')
        self.method = method

    def fit_transform(self, data, state_seq):
        num_states = len(np.unique(state_seq))
        # TICC
        if self.method == 'ticc':
            ticc = TICC(window_size=3, number_of_clusters=num_states,
                        lambda_parameter= 1e-3, beta= 2200,
                        maxIters=3, threshold=1e-4, num_proc=8)
            try:
                prediction, _ = ticc.fit_transform(data)
                prediction = prediction.astype(int)
            except KeyboardInterrupt:
                exit()
            except:
                return None
        # Time2State
        elif self.method == 'time2state':
            from downstream_methods.Time2State.time2state import Time2State
            from downstream_methods.Time2State.adapers import CausalConv_LSE_Adaper
            from downstream_methods.Time2State.clustering import DPGMM
            from downstream_methods.Time2State.default_params import params_LSE
            win_size = 512
            step = 100
            params_LSE['in_channels'] = data.shape[1]
            params_LSE['win_type'] = 'hanning' # {'hanning', 'rect'}
            params_LSE['win_size'] = win_size
            try:
                t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
                prediction = t2s.state_seq
            except KeyboardInterrupt:
                exit()
            except:
                return None
        # GHMM
        elif self.method == 'ghmm':
            ghmm = GaussianHMM(n_components=num_states, covariance_type="full", n_iter=100)
            # GMMHMM often fails to converge
            # ghmm = GMMHMM(n_components=num_states, n_mix=10, n_iter=10, covariance_type="full")
            try:
                ghmm.fit(data)
                prediction = ghmm.decode(data)[-1].astype(int)
            except KeyboardInterrupt:
                exit()
            except:
                return None
        elif self.method == 'e2usd':
            from downstream_methods.E2USD.e2usd import E2USD
            from downstream_methods.E2USD.adapers import E2USD_Adaper
            from downstream_methods.E2USD.clustering import DPGMM_E2
            from downstream_methods.E2USD.params import params
            win_size = 256
            step = 50
            params['in_channels'] = data.shape[1]
            params['compared_length'] = 256
            params['out_channels'] = 4
            try:
                t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM_E2(None)).fit(data, win_size, step)
                prediction = t2s.state_seq
            except KeyboardInterrupt:
                exit()
            except:
                return None
        elif self.method == 'clasp':
            # 2x downsampling for ClaSP
            # ClaSP has low scalability in terms of length and dimension
            num_cps = len(find_cut_points_from_state_seq(state_seq))
            # data = data[::2]
            # state_seq = state_seq[::2]
            try:
                prediction = ModifiedClaSP(data, 50, 20, num_states, 0.04)
                prediction = np.array(prediction, dtype=int)
            except KeyboardInterrupt:
                exit()
            except:
                return None
        return prediction