import logging

import attr
import numpy as np
from scipy.linalg import svd, eig

logger = logging.getLogger(__name__)


@attr.s
class KbdmInfo:
    m = attr.ib()
    l = attr.ib()
    p = attr.ib()
    q = attr.ib()
    singular_values = attr.ib()


def kbdm(data, dwell, m=None, p=1, l=None, q=0):
    """
    :param numpy.ndarray data:
        Complex input data.

    :param float dwell:
        Dwell time in seconds.

    :param int|TypeNone m:
        Number of columns/rows of U matrices.
        Default is n = len(data) / 2.

    :param int p:
        Eigenvalue exponent of the generalized eigenvalue equation. It will represent a 'shift' during the construction
        of U^p and U^{p-1} matrices.
        Default is 1.

    :param int|TypeNone l:
        ..see:: _solve_gep_svd
        If set to None, default is m.
        Default is None (m).

    :param float q:
        ..see:: _solve_gep_svd
        Default is 0.

    :return:
        # @TODO: Update.
        Spectrum Line Lists Estimations inside tuples with the following order: (Amplitude, T2, Frequency, Phase).
    :rtype: tuple(numpy.ndarray[l, 4], KbdmInfo)
    """
    if m is None and l is None:
        raise ValueError("l or m must be specified")
    elif m is None:
        m = l
    elif l is None:
        l = m
    elif l > m:
        raise ValueError("l can't be greater than m")

    m_max = (data.size + 1 - p) / 2

    if m > m_max or l > m_max:
        raise ValueError("m or l can't be greater than (n + 1 - p)/2.")

    U0, Up_1, Up = _compute_U_matrices(data=data, m=m, p=p)

    μ, B_norm, svd_info = _solve_gep_svd(U0=U0, Up_1=Up_1, Up=Up, l=l, q=q)

    info = KbdmInfo(m=m, p=p, l=l, q=q, singular_values=svd_info['singular_values'])

    # Complex amplitude calculations
    D_sqrt = data[:m] @ B_norm

    assert D_sqrt.shape == (l,)

    D = D_sqrt * D_sqrt

    # real amplitudes and phases
    A = np.abs(D)
    PH = np.angle(D)

    # Obtaining T2 and w values from eigenvalues
    Ω = -1j * np.log(μ) / dwell

    F = np.real(Ω) / (2 * np.pi)
    T2 = 1. / np.imag(Ω)

    # @TODO: this is a not a good return type. Fix and update docs.
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
    :rtype: tuple(numpy.ndarray[m,m], numpy.ndarray[m,m], numpy.ndarray[m,m])
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


def _solve_gep_svd(U0, Up_1, Up, l, q=0.0):
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

    :param float q:
        Tikhonov regularization (TR) parameter. If q = 0, TR is ignored.
        Default is 0.

    :return: computed eigenvalues (μ) and normalized eigenvectors (B)
    :rtype: tuple(numpy.ndarray[l], numpy.ndarray[m,l])
    """
    m = len(U0)

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

    assert μ.shape == (l,), f'Shape of B is {μ.shape} rather than {(m, l)}'

    # Obtaining eigenvectors at krylov basis
    # Each one of the l eigenvectors Bk with m elements is a column of B (m x l)
    B = R_ @ Dsqi_ @ P

    assert B.shape == (m, l), f'Shape of B is {B.shape} rather than {(m, l)}'

    B_norm = _normalize_eigenvectors(B, U0)

    assert B_norm.shape == (m, l), f'Shape of B_norm is {B_norm.shape} rather than {(m, l)}'

    svd_info = {
        'singular_values': s,
        'q': q,
        'l': l
    }

    return μ, B_norm, svd_info


def _normalize_eigenvectors(B, U0):
    """
    Apply normalization on each column of B matrix (B_k), such that:

    B_k^{norm} = B_k (B_k^T U0 B_k)^{-1/2}

    :param numpy.ndarray B:
        Eigenvector matrix.

    :param numpy.ndarray U0:
        U^0 matrix.
        ..see:: _compute_U_matrices

    :return: Normalized B_norm matrix, with same dimensions of B.
    :rtype: numpy.array[m,l]
    """
    # normalization factor for eigenvectors
    N_inv_sqrt = np.einsum('jk,ij,ik->k', B, U0, B)
    N_sqrt = np.sqrt(1. / N_inv_sqrt)

    # Normalized eigenvectors calculations
    B_norm = B * N_sqrt

    assert B_norm.shape == B.shape

    return B_norm
