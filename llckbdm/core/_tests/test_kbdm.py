import pytest
import pandas as pd
import numpy as np

from llckbdm.core import sig_gen
from llckbdm.core.kbdm import kbdm, _compute_U_matrices


@pytest.fixture
def N():
    return 2048


@pytest.fixture()
def dwell():
    return 5e-4


@pytest.fixture
def t(N, dwell):
    return np.linspace(0, dwell * N, N)


@pytest.fixture
def columns():
    return ['amplitude', 't2', 'frequency', 'phase']


@pytest.fixture
def df_params_brain_sim(data_path, columns):
    df = pd.read_csv(
        f'{data_path}/brain_simulated_1_5T/params.csv',
        names=columns
    )
    df = df.sort_values(['frequency'])
    return df


@pytest.fixture
def data_brain_sim(df_params_brain_sim, t):
    data = sig_gen.multi_fid(t, df_params_brain_sim.as_matrix())

    return data


def test_kbdm(data_brain_sim,  dwell, df_params_brain_sim, columns):
    m = 300

    params_est = np.column_stack(
        kbdm(
            data_brain_sim,
            dwell,
            m=m
        )
    )

    assert len(params_est) == m

    df_est = pd.DataFrame(data=params_est, columns=columns)

    df_est = df_est[df_est['amplitude'] > 1e-4]
    df_est = df_est.sort_values(['frequency'])

    assert len(df_est) == 16

    for i in range(16):
        assert pytest.approx(df_est['amplitude'].iloc[i]) == df_params_brain_sim['amplitude'].iloc[i], \
            f'Amplitude does not match at peak #{i}'

        assert pytest.approx(df_est['t2'].iloc[i], rel=1e-3) == df_params_brain_sim['t2'].iloc[i], \
            f'T2 does not match at peak #{i}'

        assert pytest.approx(df_est['frequency'].iloc[i], abs=0.3) == df_params_brain_sim['frequency'].iloc[i], \
            f'Frequency does not match at peak #{i}'

        assert pytest.approx(df_est['phase'].iloc[i], abs=1e-10) == df_params_brain_sim['phase'].iloc[i], \
            f'Phase does not match at peak #{i}'


def test_compute_U_matrices(data_brain_sim):
    m = 1000
    p = 11

    U0, Up_1, Up = _compute_U_matrices(data=data_brain_sim, m=m, p=p)

    # Validate first line of each Matrix
    assert U0[0, :m] == pytest.approx(data_brain_sim[:m])
    assert Up_1[0, :m] == pytest.approx(data_brain_sim[p - 1:m + p - 1])
    assert Up[0, :m] == pytest.approx(data_brain_sim[p:m + p])

    # Validate last line of each Matrix
    assert U0[m - 1, :m] == pytest.approx(data_brain_sim[m - 1:2 * m - 1])
    assert Up_1[m - 1, :m] == pytest.approx(data_brain_sim[m + p - 2:2 * m + p - 2])
    assert Up[m - 1, :m] == pytest.approx(data_brain_sim[m + p - 1:2 * m + p - 1])