import pytest
import numpy as np
import pandas as pd

from llckbdm.kbdm import kbdm, _compute_U_matrices


def test_kbdm_svd(data_brain_sim, dwell, df_params_brain_sim, columns):
    m = 300

    params_est, info = kbdm(
        data_brain_sim,
        dwell,
        m=m,
    )

    assert params_est.shape == (m, 4)

    assert info.m == m
    assert info.l == m
    assert info.q == pytest.approx(0)
    assert info.p == 1

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
    m = 300
    p = 2

    U0, Up_1, Up = _compute_U_matrices(data=data_brain_sim, m=m, p=p)

    # Validate first line of each Matrix
    assert U0[0, :m] == pytest.approx(data_brain_sim[:m])
    assert Up_1[0, :m] == pytest.approx(data_brain_sim[p - 1:m + p - 1])
    assert Up[0, :m] == pytest.approx(data_brain_sim[p:m + p])

    # Validate last line of each Matrix
    assert U0[m - 1, :m] == pytest.approx(data_brain_sim[m - 1:2 * m - 1])
    assert Up_1[m - 1, :m] == pytest.approx(data_brain_sim[m + p - 2:2 * m + p - 2])
    assert Up[m - 1, :m] == pytest.approx(data_brain_sim[m + p - 1:2 * m + p - 1])


def test_kbdm_should_check_for_m_and_l_values(data_brain_sim, dwell):
    with pytest.raises(ValueError) as e_info:
        kbdm(data=data_brain_sim, dwell=dwell)

    assert "l or m must be specified" in str(e_info.value)

    with pytest.raises(ValueError) as e_info:
        kbdm(data=data_brain_sim, dwell=dwell, l=30, m=20)

    assert "l can't be greater than m" in str(e_info.value)

    _, info = kbdm(data=data_brain_sim, dwell=dwell, l=30)
    assert info.l == 30
    assert info.m == 30

    _, info = kbdm(data=data_brain_sim, dwell=dwell, m=30)
    assert info.l == 30
    assert info.m == 30


def test_kbdm_m_none_should_use_default_value(data_brain_sim, dwell):
    l = 30
    line_lists, info = kbdm(data=data_brain_sim, dwell=dwell, l=l)

    assert np.shape(line_lists) == (l, 4)
    assert info.m == l
    assert info.l == l


def test_kbdm_invalid_m_n_p_constraint_should_raise_value_error(data_brain_sim, dwell):
    with pytest.raises(ValueError) as e_info:
        kbdm(data=data_brain_sim, dwell=dwell, m=int(len(data_brain_sim)/2) + 1)

    assert "m or l can't be greater than (n + 1 - p)/2." in str(e_info.value)

    with pytest.raises(ValueError) as e_info:
        kbdm(data=data_brain_sim, dwell=dwell, l=int(len(data_brain_sim)/2) + 1, m=int(len(data_brain_sim)/2) + 1)

    with pytest.raises(ValueError) as e_info:
        kbdm(data=data_brain_sim, dwell=dwell, l=10, m=int(len(data_brain_sim)/2) + 1)

    assert "m or l can't be greater than (n + 1 - p)/2." in str(e_info.value)


def test_kbdm_svd_with_q_greater_than_zero_should_use_tikhonov_regularization(data_brain_sim, dwell, caplog):
    caplog.set_level('DEBUG')

    m = 10

    params_est, info = kbdm(
        data_brain_sim,
        dwell,
        m=m,
        q=1e-3,
    )

    assert 'Using Tikhonov Regularization' in caplog.text

    assert params_est.shape == (m, 4)
    assert info.q == pytest.approx(1e-3)
    assert info.m == m
