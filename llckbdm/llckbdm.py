import logging

import hdbscan
import attr
import numpy as np

from llckbdm.min_rmse_kbdm import min_rmse_kbdm
from llckbdm.sampling import sample_kbdm, filter_samples

logger = logging.getLogger(__name__)

@attr.s
class LlcKbdmResult:
    line_list = attr.ib()
    rmse = attr.ib()


def llc_kbdm(data, dwell, m_range, gep_solver='svd', p=1, l=None, q=None):
    """
    Compute Line List Clustering Krylov Basis Diagonalization Method (LLC-KBDM).

    :param numpy.ndarray data:
        Complex input data.

    :param float dwell:
        Dwell time in seconds.

    :param list|range m_range:
        Range or list with number of columns/rows of U matrices for iteration of KBDM sampling.
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

    :param float q:
        This is used only with if gep_solver is set to 'svd'.
        ..see:: llckbdm.kbdm._solve_gep_svd
        Default is 0.

    :return: @TODO
    :rtype: @TODO
    """
    if len(m_range) < 2:
        raise ValueError("size of 'm_range' must be greater than 2.")

    # KBDM sampling for m values inside m_range
    line_lists, infos = sample_kbdm(
        data=data,
        dwell=dwell,
        m_range=m_range,
        p=p,
        gep_solver=gep_solver,
        l=l,
        q=q,
    )

    # concatenate all sampled line lists into a single dataset
    samples = np.concatenate(line_lists)

    samples = filter_samples(samples)

    transf_line_list = _transform_line_lists(samples, dwell)

    summarized_line_lists = []

    m_range_size = len(m_range)

    for min_samples in range(int(np.ceil(0.10 * m_range_size )), m_range_size):
        logger.debug('HDBSCAN with min_samples = %d', min_samples)

        num_clusters, labels, clustered, non_clustered = _cluster_line_lists(
            data=data,
            samples=transf_line_list,
            eps=0.01,
            min_samples=min_samples
        )

        summarized_line_list = _summarize_clusters(
            samples=samples,
            clusters=clustered,
        )

        summarized_line_lists.append(summarized_line_list)

    min_rmse_kbdm_results = min_rmse_kbdm(data=data, dwell=dwell, samples=summarized_line_lists)

    return LlcKbdmResult(
        line_list=min_rmse_kbdm_results.line_list,
        rmse=min_rmse_kbdm_results.min_rmse
    )


def _transform_line_lists(line_lists, dwell):
    """
    Transform line list data space into a basis composed by real and imaginary partes of eigenvalues of GEP problem;
    amplitude and phase;

    :param line_lists:
        Features obtained from KBDM computations.

    :param dwell:
        ..see:: llc_kbdm

    :return: Array containing features at the eigenvalues basis.
    :rtype: numpy.ndarray
    """
    A = line_lists[:, 0]
    T2 = line_lists[:, 1]
    F = line_lists[:, 2]
    PH = line_lists[:, 3] * 0

    OMEGA = 2 * np.pi * F + 1j / T2

    mu = np.exp(1j * dwell * OMEGA)

    MU_RE = np.real(mu)
    MU_IM = np.imag(mu)

    return np.column_stack(
        (MU_RE, MU_IM, A, PH)
    )


def _inverse_transform_line_lists(transformed_line_lists, dwell):
    """
    Apply inverse transformation of line lists into original data space basis composed by amplitude,
    transversal relaxation time, frequency and phase.

    :param numpy.ndarray transformed_line_lists:
        Array containing transformed features.

    :param dwell:
        ..see:: llc_kbdm

    :return: Array containing features at the canonical data basis.
    :rtype: numpy.ndarray
    """
    MU_RE = transformed_line_lists[:, 0]
    MU_IM = transformed_line_lists[:, 1]
    A = transformed_line_lists[:, 2]
    PH = transformed_line_lists[:, 3]

    MU = MU_RE + 1j * MU_IM

    OMEGA = -1j * np.log(MU) / dwell

    T2 = 1. / np.imag(OMEGA)
    F = np.real(OMEGA) / (2 * np.pi)

    return np.column_stack(
        (A, T2, F, PH)
    )


def _cluster_line_lists(data, samples, eps, min_samples):
    """
    Use DBSCAN to cluster samples.

    :param numpy.ndarray samples:
        Array containing line lists to be clustered

    :param eps:
        ..see:: sklearn.cluster.DBSCAN

    :param min_samples:
        ..see:: sklearn.cluster.DBSCAN

    :return: number of estimated clusters, labels for each sample (with same dimensions of samples input),
        list containing labels of each cluster and labels of non-clustered samples.

    :rtype: tuple(int, numpy.ndarray, list(tuple), tuple)
    """

    #cl = DBSCAN(eps=eps, min_samples=min_samples)
    cl = hdbscan.HDBSCAN(min_samples=min_samples)

    cl.fit(samples)
    labels = cl.labels_

    num_clusters = len(set(labels) - {-1})

    clustered = []

    for cluster_label in range(num_clusters):
        clustered.append(np.nonzero(labels == cluster_label))

    non_clustered = np.nonzero(labels == -1)

    return num_clusters, labels, clustered, non_clustered


def _summarize_clusters(samples, clusters, summarizer=None):
    """
    Compute compact line list from

    :param numpy.ndarray samples:
        KBDM line listed sampled in a given range of m values.

    :param list clusters:
        List containing indexes of each cluster.

    :param function summarizer:
        Callback function for summarizing each cluster into a single feature.

    :return:
    """
    line_list = []

    if summarizer is None:
        #summarizer = np.average
        summarizer = np.median

    for cluster in clusters:
        cluster_samples = samples[cluster]
        cluster_samples[:,1] = 1 / cluster_samples[:,1]
        summarized_cluster = summarizer(cluster_samples, axis=0)
        summarized_cluster[1] = 1 / summarized_cluster[1]
        # summarized_cluster = np.average(cluster_samples, axis=0)

        line_list.append(summarized_cluster)

    return np.array(line_list)
