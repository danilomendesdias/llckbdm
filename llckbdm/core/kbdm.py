import numpy as np
from scipy.linalg import svd, eig


def _compute_U_matrices(data, m, p):
    """
    Compute U^0, U^{p-1} and U^p matrices.

    :param numpy.ndarray data:
        Complex array containing the input signal in the time domain.

    :param int m:
        Number of columns/rows of U matrices.

    :param int p:
        Eigenvalue power parameter of the generalized eigenvalue equation. It will represent a 'shift' on data
        array during the construction of U^p and U^{p-1} matrices.

    :return: U^0, U^{p-1} and U^p matrices, each one with dimensionality m x m.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    U0 = np.empty((m, m), dtype=np.complex)
    Up_1 = np.empty((m, m), dtype=np.complex)
    Up = np.empty((m, m), dtype=np.complex)

    for i in range(m):
        U0[i] = data[i:i + m]  # U^0
        Up[i] = data[i + p:i + m + p]  # U^p

    if p != 1:
        for i in range(m):
            Up_1[i] = data[i + p - 1:i + m + p - 1]  # U^{p-1}
    else:
        Up_1 = np.copy(U0)

    return U0, Up_1, Up


def _solve_gep_svd(U0, Up_1, Up, **kwargs):
    # Tikhonov regularization parameter
    q = kwargs.get("q", 0)

    m = len(U0)

    # Number of singular components
    l = kwargs.get("l", m)

    if l > m and l != "auto":
        raise ValueError("l can't be greater than m.")

    # Decomposing U0 into singular values.
    # _h suffix denotes hermitian operator.
    L, s, R_h = svd(Up_1)

    # Minimal energy % for SVD reduction (only works with l = 'auto')
    # @TODO: check relevance of this approach
    e = kwargs.get("e", 0.90)

    if l == "auto":
        s = trunc_d(m, s, e)
        l = len(s)

    S = np.diag(s)

    # reduced matrices
    L_ = L[:, :l]
    S_ = S[:l, :l]
    Rh_ = R_h[:l, :]

    # apply hermitian operators
    L_h_ = L_.conj().transpose()
    R_ = Rh_.conj().transpose()

    if q > 0:
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
    B_norm = _normalize_eigenvector(B, U0)

    return μ, B_norm


def _normalize_eigenvector(B, U0):
    # normalization factor for eigenvectors
    N_inv_sqrt = np.einsum('jk,ij,ik->k', B, U0, B)
    N_sqrt = np.sqrt(1./N_inv_sqrt)

    # Normalized eigenvectors calculations
    B_norm = (B * N_sqrt).T

    return B_norm


def _solve_gep(U0, Up_1, Up):
    μ, B = eig(b=Up_1, a=Up, overwrite_a=True, overwrite_b=True)

    B_norm = _normalize_eigenvector(B, U0)

    return μ, B_norm


def kbdm(data, dwell, gep_solver='svd', **kwargs):
    """
    :param numpy.ndarray data:
        Input Data
    :param float dwell:
        Dwell time in seconds.

    :return:
        Spectrum Line Lists Estimations.
    :rtype: numpy.ndarray
    """
    p = kwargs.get("p", 1)
    m = kwargs.get("m", int((data.size + 1 - p) / 2))

    if m > (data.size + 1 - p) / 2:
        raise ValueError("m can't be greater than (n + 1 - p)/2.")

    U0, Up_1, Up = _compute_U_matrices(data=data, m=m, p=p)

    assert gep_solver in ['svd', 'scipy'], "GEP solver can be 'svd' or 'scipy'"

    if gep_solver == 'svd':
        μ, B_norm = _solve_gep_svd(U0=U0, Up_1=Up_1, Up=Up, **kwargs)
    else:
        μ, B_norm = _solve_gep(U0=U0, Up_1=Up_1, Up=Up)

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

    return A, T2, F, PH


def trunc_d(m, d, e):
    daux = d / d[0]
    d0 = daux[0]
    exp = np.ceil(np.log10(d0))
    exp_min = exp - e
    min_ = 10 ** exp_min
    
    d = d[np.where(d / d[0] > min_)[0]]
    
    return d

def auto_l_percentage(m, d, e):
    l = m
    
    d2 = d
    etotal = np.sum(d2**2) # energy
    emin = etotal * e
    
    ecurr = etotal

    for s2 in d2[::-1]:
        ecurr = ecurr - s2
        if ecurr < emin:
            break
        l = l - 1        
    
    print("auto_l:", l)
    
    return l
