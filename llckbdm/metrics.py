import numpy as np
from sklearn.metrics import mean_squared_error

from llckbdm.sig_gen import gen_t_freq_arrays, multi_fid


def calculate_freq_domain_rmse(data, params_est, dwell):
    N = len(data)

    t_array, freq_array = gen_t_freq_arrays(N=N, dwell=dwell)

    data_est = multi_fid(t_array=t_array, params=params_est)

    data_est_fft = np.fft.fft(data_est) / np.sqrt(N)
    data_fft = np.fft.fft(data) / np.sqrt(N)

    return np.sqrt(mean_squared_error(data_fft.real, data_est_fft.real))
