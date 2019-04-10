import pytest
import numpy as np
import pandas as pd
from scipy.linalg import svd, eig

from llckbdm._bindings import (
    kbdm as kbdm_cpp, _compute_U_matrices, _solve_gep_svd as _solve_gep_svd_cpp,
    _normalize_eigenvectors as _normalize_eigenvectors_cpp,
    _solve_svd,
    _solve_eig,
    _compute_U,
    _compute_B,
    _compute_reduced_matrices
)
from llckbdm.kbdm import _solve_gep_svd, _normalize_eigenvectors, kbdm


@pytest.fixture
def m():
    return 400


@pytest.fixture
def l():
    return 30


def test_kbdm_cpp(data_brain_sim, m, l, dwell, df_params_brain_sim):
    A, T2, F, PH = kbdm_cpp(data_brain_sim, dwell, m, l, 1)

    df_params_estimated = pd.DataFrame(
        data=np.column_stack((A, T2, F, PH)),
        columns=df_params_brain_sim.columns
    )

    df_params_estimated = df_params_estimated.query('amplitude > 1e-3')
    df_params_estimated.sort_values(by=['frequency'], inplace=True)

    assert df_params_estimated.values == pytest.approx(df_params_brain_sim.values)


@pytest.mark.xfail(reason="To be investigated")
def test_solve_gep_svd(data_brain_sim, m, l):
    p = 1

    U0, Up_1, Up = _compute_U_matrices(data_brain_sim, m, p)
    expected_mu, expected_B, _ = _solve_gep_svd(U0, Up_1, Up, l)

    mu, B = _solve_gep_svd_cpp(U0, Up_1, Up, l)

    assert mu.shape == (l,)
    assert B.shape == (m, l)

    for k in range(l):
        Bk = B[:, k]
        assert Bk @ Up == pytest.approx(mu[k] * Bk @ U0)


def test_compute_U_matrices(data_brain_sim, m):
    p = 1

    U0, Up_1, Up = _compute_U_matrices(data_brain_sim, m, p)

    # Validate first line of each Matrix
    assert U0[0, :m] == pytest.approx(data_brain_sim[:m])
    assert Up_1[0, :m] == pytest.approx(data_brain_sim[p - 1:m + p - 1])
    assert Up[0, :m] == pytest.approx(data_brain_sim[p:m + p])

    # Validate last line of each Matrix
    assert U0[m - 1, :m] == pytest.approx(data_brain_sim[m - 1:2 * m - 1])
    assert Up_1[m - 1, :m] == pytest.approx(data_brain_sim[m + p - 2:2 * m + p - 2])
    assert Up[m - 1, :m] == pytest.approx(data_brain_sim[m + p - 1:2 * m + p - 1])


def test_compute_reduced_matrices(m, l):
    L = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    s = np.sort(np.random.randn(m) ** 2)[::-1]
    S = np.diag(s)
    R = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    R_h = np.conjugate(R).transpose()

    # reduced matrices
    L_ = L[:, :l]
    S_ = S[:l, :l]
    Rh_ = R_h[:l, :]

    # apply hermitian operators
    expected_L_h_ = L_.conj().transpose()
    expected_R_ = Rh_.conj().transpose()
    expected_Dsqi_ = np.linalg.inv(np.sqrt(S_))

    Dsqi_, L_h_, R_ = _compute_reduced_matrices(L, s, R, m, l)

    assert L_h_.shape == expected_L_h_.shape == (l, m)
    assert Dsqi_.shape == expected_Dsqi_.shape == (l, l)
    assert R_.shape == expected_R_.shape == (m, l)

    assert Dsqi_ == pytest.approx(expected_Dsqi_)
    assert L_h_ == pytest.approx(expected_L_h_)
    assert R_ == pytest.approx(expected_R_)


def test_solve_svd(m):
    Up_1 = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    expected_L, expected_s, expected_R_h = svd(Up_1)

    expected_R = np.conjugate(expected_R_h).transpose()

    L, s, R = _solve_svd(Up_1)

    assert s.shape == (m,)
    assert L.shape == (m, m)
    assert R.shape == (m, m)

    assert expected_L @ np.conjugate(expected_L).transpose() == pytest.approx(np.identity(m))

    assert s == pytest.approx(expected_s)

    assert np.abs(L) == pytest.approx(np.abs(expected_L))
    assert np.abs(R) == pytest.approx(np.abs(expected_R))

    assert L @ np.diag(s) @ np.conjugate(R).transpose() == pytest.approx(Up_1)


def test_compute_U(m, l):
    Dsqi_ = np.sort(np.random.randn(l, l)**2)
    L_h_ = np.random.randn(l, m) + 1j * np.random.randn(l, m)
    R_ = np.random.randn(m, l) + 1j * np.random.randn(m, l)
    Up = np.random.randn(m, m) + 1j * np.random.randn(m, m)

    expected_U = Dsqi_ @ L_h_@ Up @ R_@ Dsqi_

    U = _compute_U(Dsqi_, L_h_, Up, R_)

    assert U.shape == expected_U.shape == (l, l)
    assert U == pytest.approx(expected_U)


def test_solve_eig(l):
    U = np.random.randn(l, l) + 1j * np.random.randn(l, l)

    expected_mu, expected_P = eig(U)
    mu, P =_solve_eig(U)

    order = np.argsort(mu)
    expected_order = np.argsort(expected_mu)

    mu = mu[order]
    P = P[:, order]

    expected_mu = expected_mu[expected_order]
    expected_P = expected_P[:, expected_order]

    assert P.shape == (l, l)
    assert mu.shape == (l,)

    assert expected_mu == pytest.approx(mu)
    assert P @ np.diag(mu) @ np.linalg.inv(P) == pytest.approx(U)
    assert expected_P @ np.diag(expected_mu) @ np.linalg.inv(expected_P) == pytest.approx(U)


def test_compute_B(m, l):
    R_ = np.random.randn(m, l) + 1j * np.random.randn(m, l)
    Dsqi_ = np.sort(np.random.randn(l, l)**2)
    P = np.random.randn(l, l) + 1j * np.random.randn(l, l)

    B = _compute_B(R_, Dsqi_, P)

    assert B.shape == (m, l)
    assert B == pytest.approx(R_ @ Dsqi_ @ P)


def test_normalize_eigenvectors(data_brain_sim, m, l):
    m, l = 4, 8
    B = np.random.randn(m, l) + 1j * np.random.randn(m, l)
    U0 = np.random.randn(m, m) + 1j * np.random.randn(m, m)

    expected_values = _normalize_eigenvectors(B, U0)

    assert _normalize_eigenvectors_cpp(B, U0).shape == expected_values.shape == (m, l)
    assert _normalize_eigenvectors_cpp(B, U0) == pytest.approx(expected_values)
