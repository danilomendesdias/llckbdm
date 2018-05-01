import pytest
import numpy as np
from sklearn.metrics import mean_squared_error

from llckbdm.metrics import calculate_freq_domain_rmse


def test_calculate_freq_domain_rmse(data_brain_sim, dwell, params_brain_sim, N):
    rmse = calculate_freq_domain_rmse(data=data_brain_sim, params_est=params_brain_sim, dwell=dwell)
    # testing output without noise
    assert rmse == pytest.approx(0)

    # including noise at real contribution only (where it will be computed)
    noise = np.random.randn(N) + 1j * np.random.randn(N)
    data_with_noise = data_brain_sim + noise

    data_fft_raw = np.fft.fft(data_brain_sim) / np.sqrt(N)
    data_fft_with_noise = np.fft.fft(data_with_noise) / np.sqrt(N)

    noisy_rmse = calculate_freq_domain_rmse(data=data_with_noise, params_est=params_brain_sim, dwell=dwell)
    assert noisy_rmse == pytest.approx(np.sqrt(mean_squared_error(data_fft_with_noise.real, data_fft_raw.real)))

