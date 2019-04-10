import pytest
import pandas as pd
import numpy as np

from llckbdm import sig_gen


@pytest.fixture
def N():
    return 2048


@pytest.fixture()
def dwell():
    return 5e-4


@pytest.fixture
def t(N, dwell):
    return np.linspace(0, dwell * N, N, endpoint=False)


@pytest.fixture
def columns():
    return ['amplitude', 't2', 'frequency', 'phase']


@pytest.fixture
def df_params_brain_sim(data_path, columns):
    df = pd.read_csv(
        f'{data_path}/params_brain_sim_1_5T.csv',
        names=columns
    )
    df = df.sort_values(['frequency'])
    return df


@pytest.fixture
def params_brain_sim(df_params_brain_sim):
    return df_params_brain_sim.values


@pytest.fixture
def data_brain_sim(df_params_brain_sim, t):
    data = sig_gen.multi_fid(t, df_params_brain_sim.values)

    return data


@pytest.fixture
def t_array(N, dwell):
    return np.linspace(0, dwell * N, N, endpoint=False)


@pytest.fixture
def freq_array(N, dwell):
    return np.fft.fftshift(np.fft.fftfreq(N, dwell))
