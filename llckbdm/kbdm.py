import logging

import numpy as np
from scipy.linalg import svd, eig

logger = logging.getLogger(__name__)


def kbdm(data, dwell, m=None, gep_solver='svd', p=1, l=None, q=0):
    """
    :param numpy.ndarray data:
        Complex input data.

    :param float dwell:
        Dwell time in seconds.

    :param int m:
        Number of columns/rows of U matrices.

    :param str gep_solver:
        Method used to solve Generalized Eigenvalue Problem. Can be 'svd', for self-implemented solution; or scipy
        to use eig function from scipy.linalg.eig.
        Default is 'svd'.

    :param int p:
        Eigenvalue exponent of the generalized eigenvalue equation. It will represent a 'shift' during the construction
        of U^p and U^{p-1} matrices.

    :param int l:
        This is used only with if gep_solver is set to 'svd'.
        ..see:: _solve_gep_svd
        Default is None.

    :param int q:
        This is used only with if gep_solver is set to 'svd'.
        ..see:: _solve_gep_svd
        Default is 0.

    :return:
        Spectrum Line Lists Estimations inside tuples with the following order: (Amplitude, T2, Frequency, Phase).
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    if m is None:
        m = int((data.size + 1 - p) / 2)
    elif m > (data.size + 1 - p) / 2:
        raise ValueError("m can't be greater than (n + 1 - p)/2.")

    if gep_solver not in ['svd', 'scipy']:
        raise ValueError("GEP solver can be 'svd' or 'scipy'")

    U0, Up_1, Up = _compute_U_matrices(data=data, m=m, p=p)

    # @TODO: use attrs?
    info = {
        'm': m,
        'p': p,
    }

    if gep_solver == 'svd':
        μ, B_norm, svd_info = _solve_gep_svd(U0=U0, Up_1=Up_1, Up=Up, l=l, q=q)

        info['q'] = svd_info['q']
        info['l'] = svd_info['l']
    else:
        μ, B_norm = _solve_gep_scipy(U0=U0, Up_1=Up_1, Up=Up)

    # Complex amplitude calculations
    D_sqrt = B_norm @ data[:m]
    D = D_sqrt * D_sqrt

    # real amplitudes and phases
    A = np.abs(D)
    PH = np.angle(A)

    # Obtaining T2 and w values from eigenvalues
    Ω = -1j * np.log(μ) / dwell

    F = np.real(Ω) / (2 * np.pi)
    T2 = 1. / np.imag(Ω)

    line_list = np.column_stack(
        (A, T2, F, PH,)
    )

    return line_list, info


def _compute_U_matrices(data, m, p):
    """
    Compute U^0, U^{p-1} and U^p matrices.

    :param numpy.ndarray data:
        Complex array containing the input signal in the time domain.

    :param int m:
        ..see:: kbdm

    :param int p:
        ..see:: kbdm

    :return: U^0, U^{p-1} and U^p matrices, each one with dimensionality m x m.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    U0 = np.empty((m, m), dtype=np.complex)
    Up_1 = np.empty((m, m), dtype=np.complex)
    Up = np.empty((m, m), dtype=np.complex)

    for i in range(m):
        logger.debug('Constructing U0')
        U0[i] = data[i:i + m]  # U^0

        logger.debug('Constructing Up')
        Up[i] = data[i + p:i + m + p]  # U^p

    logger.debug('Constructing Up_1')
    if p == 1:
        logger.debug('As p = 1, U0 will be copied to construct Up_1.')
        Up_1 = np.copy(U0)
    else:
        for i in range(m):
            Up_1[i] = data[i + p - 1:i + m + p - 1]  # U^{p-1}

    return U0, Up_1, Up


def _solve_gep_svd(U0, Up_1, Up, l=None, q=0):
    """
    Solve Generalized Eigenvalue Problem (GEP) by reducing it into an ordinary eigenvalue problem through
    Singular Value Decomposition (SVD) of U^{p-1} matrix.

    :param numpy.ndarray U0:
        U^0 matrix.
        ..see::  _compute_U_matrices

    :param numpy.ndarray Up_1:
        U^{p-1} matrix.
        ..see::  _compute_U_matrices

    :param numpy.ndarray Up:
        U^p matrix.
        ..see::  _compute_U_matrices

    :param int l:
        U matrices dimensionality are reduced to l x l after applying SVD.
        Default is len(U0), which is m.
        ..see::  _compute_U_matrices

    :param int q:
        Tikhonov regularization (TR) parameter. If q = 0, TR is ignored.
        Default is 0.

    :return: computed eigenvalues (μ) and normalized eigenvectors (B)
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    m = len(U0)

    if q is None:
        q = 0

    if l is None:
        l = m

    if l > m:
        raise ValueError("l can't be greater than m.")

    # Decomposing U0 into singular values.
    # _h suffix denotes hermitian operator.
    L, s, R_h = svd(Up_1)

    S = np.diag(s)

    # reduced matrices
    L_ = L[:, :l]
    S_ = S[:l, :l]
    Rh_ = R_h[:l, :]

    # apply hermitian operators
    L_h_ = L_.conj().transpose()
    R_ = Rh_.conj().transpose()

    if q > 0:
        logger.debug('Using Tikhonov Regularization with q=%f', q)
        # including tikhonov regularization
        S = S_ + q * q * np.linalg.inv(S_)

        Dsqi_ = np.linalg.inv(np.sqrt(S))
    else:
        Dsqi_ = np.linalg.inv(np.sqrt(S_))

    # U computation
    U = Dsqi_ @ L_h_ @ Up @ R_ @ Dsqi_

    # U diagonalization (mu are the final eigenvalues!)
    μ, P = eig(U)

    # Obtaining eigenvectors at krylov basis
    B = R_ @ Dsqi_ @ P
    B_norm = _normalize_eigenvectors(B, U0)

    svd_info = {
        'singular_values': s,
        'q': q,
        'l': l
    }

    return μ, B_norm, svd_info


def _solve_gep_scipy(U0, Up_1, Up):
    """
    Solve the same equation described by _solve_gep_svd, but using scipy.linalg.eig implementation.

    ..see:: scipy.linalg.eig for implementation details
    ..see::  _solve_gep_svd for arguments documentation

    :return: computed eigenvalues (μ) and normalized eigenvectors (B)
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    μ, B = eig(b=Up_1, a=Up, overwrite_a=True, overwrite_b=True)

    B_norm = _normalize_eigenvectors(B, U0)

    return μ, B_norm


def _normalize_eigenvectors(B, U0):
    """
    Apply normalization on each column of B matrix (B_k), such that:

    B_k^{norm} = B_k (B_k^T U0 B_k)^{-1}

    :param numpy.ndarray B:
        Eigenvector matrix.

    :param numpy.ndarray U0:
        U^0 matrix.
        ..see:: _compute_U_matrices

    :return:
    """
    # normalization factor for eigenvectors
    N_inv_sqrt = np.einsum('jk,ij,ik->k', B, U0, B)
    N_sqrt = np.sqrt(1./N_inv_sqrt)

    # Normalized eigenvectors calculations
    B_norm = (B * N_sqrt).T

    return B_norm
