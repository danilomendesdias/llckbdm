import logging

import numpy as np

from llckbdm.kbdm import kbdm

logger = logging.getLogger(__name__)


def llc_kbdm(data, dwell, m_range, gep_solver='svd', p=1, l=None, q=None):
    """
    Compute Line List Clusternig Krylov Basis Diagonalization Method (LLC-KBDM).

   :param numpy.ndarray data:
        Complex input data.

    :param float dwell:
        Dwell time in seconds.

    :param list|range m_range:
        Range or list with number of columns/rows of U matrices on each KBDM sample.
        Its size must be greater than 2.

    :param str gep_solver:
        Method used to solve Generalized Eigenvalue Problem. Can be 'svd', for self-implemented solution; or scipy
        to use eig function from scipy.linalg.eig.
        Default is 'svd'.

    :param int p:
        Eigenvalue exponent of the generalized eigenvalue equation. It will represent a 'shift' during the construction
        of U^p and U^{p-1} matrices.

    :param int l:
        This is used only with if gep_solver is set to 'svd'.
        ..see:: llckbdm.kbdm._solve_gep_svd
        Default is None.

    :param int q:
        This is used only with if gep_solver is set to 'svd'.
        ..see:: llckbdm.kbdm._solve_gep_svd
        Default is 0.

    :return: @TODO
    :rtype: @TODO
    """
    if len(m_range) < 2:
        raise ValueError("size of 'm_range' must be greater than 2.")

    # KBDM sampling for m values inside m_range
    line_lists, infos = _sample_kbdm(
        data=data,
        dwell=dwell,
        m_range=m_range,
        p=p,
        gep_solver=gep_solver,
        l=l,
        q=q,
    )

    transf_line_list = _transform_line_lists(line_lists, dwell)

    clusters = _cluster_line_lists(transf_line_list)

    final_line_lists = _inverse_transform_line_lists(clusters)

    return final_line_lists


def _sample_kbdm(data, dwell, m_range, p, gep_solver, l, q):
    """
    Compute samples of multiple computations of KBDM in a given range of m values.

    :param data:line_lists
        ..see:: llc_kbdm

    :param dwell:
        ..see:: llc_kbdm

    :param m_range:
        ..see:: llc_kbdm

    :param p:
        ..see:: llc_kbdm

    :param gep_solver:
        ..see:: llc_kbdm

    :param l:
        ..see:: llc_kbdm

    :param q:
        ..see:: llc_kbdm

    :return: @TODO
    :rtype: @TODO
    """
    line_lists = []
    infos = []

    for m in m_range:
        logger.info(f'Computing KBDM with m = {m}')

        line_list, info = kbdm(
            data=data,
            dwell=dwell,
            m=m,
            p=p,
            gep_solver=gep_solver,
            l=l,
            q=q,
        )

        line_lists.append(line_list)
        infos.append(info)

    return line_lists, infos


def _transform_line_lists(line_lists, dwell):
    A = line_lists[:, 0]
    T2 = line_lists[:, 1]
    F = line_lists[:, 2]
    PH = line_lists[:, 3]

    OMEGA = 2 * np.pi * F + 1j / T2

    mu = np.exp(1j * dwell * OMEGA)

    MU_RE = np.real(mu)
    MU_IM = np.imag(mu)

    return np.column_stack(
        (MU_RE, MU_IM, A, PH)
    )


def _inverse_transform_line_lists(transformed_line_lists, dwell):
    MU_RE = transformed_line_lists[:, 0]
    MU_IM = transformed_line_lists[:, 1]
    A = transformed_line_lists[:, 2]
    PH = transformed_line_lists[:, 3]

    MU = MU_RE  + 1j * MU_IM

    OMEGA = -1j * np.log(MU) / dwell

    T2 = 1. / np.imag(OMEGA)
    F = np.real(OMEGA) / (2 * np.pi)

    return np.column_stack(
        (A, T2, F, PH)
    )


def _cluster_line_lists(line_lists):
    pass
