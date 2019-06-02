import logging

import hdbscan
import attr
import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans, OPTICS
from llckbdm.sig_gen import multi_fid, gen_t_freq_arrays
from llckbdm.min_rmse_kbdm import min_rmse_kbdm
from llckbdm.sampling import sample_kbdm, filter_samples
from llckbdm.metrics import calculate_freq_domain_rmse

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class LlcKbdmResult:
    line_list: list = []
    rmse: float = None
    silhouette: list = []


@attr.s(auto_attribs=True)
class IterativeLlcKbdmResult:
    line_list: np.ndarray = np.array([])
    line_lists: list = []
    rmse: float = None
    silhouettes: list = []


@attr.s(auto_attribs=True)
class ClusteringResult:
    num_clusters: int = 0
    labels: np.ndarray = np.array([])
    clustered: list = []
    non_clustered: list = []
    summarized_line_list: np.ndarray = np.array([])
    clustered_silhouettes: list = []


def llc_kbdm(data, dwell, m_range, p=1, l=None, q=0.0, method='hdbscan', optics_filter=False):
    """
    Compute Line List Clustering Krylov Basis Diagonalization Method (LLC-KBDM).

    :param numpy.ndarray data:
        Complex input data.

    :param float dwell:
        Dwell time in seconds.

    :param list|range m_range:
        Range or list with number of columns/rows of U matrices for iteration of KBDM sampling.
        Its size must be greater than 2.

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

    :param str method:
        Clustering method. Can be 'hdbscan' or 'k-means'.

    :param bool optics_filter:
        Enable clustering pre-filter with OPTICS algorithm. It will remove all dataset features that are not labelled
        with OPTICS (considered as noise). Recommended with K-means method.

    :return: An object containing final estimations, RMSE values and silhouettes.
    :rtype: LlcKbdmResult
    """
    if len(m_range) < 2:
        raise ValueError("size of 'm_range' must be greater than 2.")

    # KBDM sampling for m values inside m_range
    line_lists, infos = sample_kbdm(
        data=data,
        dwell=dwell,
        m_range=m_range,
        p=p,
        l=l,
        q=q,
    )

    # concatenate all sampled line lists into a single dataset
    samples = np.concatenate(line_lists)

    samples = filter_samples(samples)

    transformed_line_list = _transform_line_lists(samples, dwell)

    if optics_filter:
        samples, transformed_line_list = filter_with_optics(samples, transformed_line_list)

    m_range_size = len(m_range)

    clustering_results = []

    if method == 'hdbscan':
        optimized_parameter = 'min_samples'
        parameter_range = range(int(np.ceil(0. * m_range_size + 1)), m_range_size)
    else:
        assert method == 'k-means', f'Invalid method: {method}'

        optimized_parameter = 'n_clusters'
        parameter_range = range(2, np.max(m_range))

    for optimized_parameter_value in parameter_range:
        clustering_result = _cluster_line_lists(
            samples=samples,
            transformed_samples=transformed_line_list,
            method=method,
            method_parameters={optimized_parameter: optimized_parameter_value}
        )

        if method == 'k-means':
            # checking whether the number of clusters trial reached the plateau
            if clustering_result.num_clusters < optimized_parameter_value:
                break

        if clustering_result.num_clusters > 0:
            clustering_results.append(clustering_result)

    summarized_line_lists = [
        cl_result.summarized_line_list for cl_result in clustering_results
    ]

    min_rmse_kbdm_results = min_rmse_kbdm(
        data=data,
        dwell=dwell,
        samples=summarized_line_lists
    )

    if min_rmse_kbdm_results is None:
        return LlcKbdmResult(
            line_list=[],
            rmse=[],
            silhouette=[],
        )

    silhouette = np.array(
        clustering_results[min_rmse_kbdm_results.min_index].clustered_silhouettes
    )

    return LlcKbdmResult(
        line_list=min_rmse_kbdm_results.line_list,
        rmse=min_rmse_kbdm_results.min_rmse,
        silhouette=silhouette
    )


def iterative_llc_kbdm(
        data, dwell, m_range, p=1, l=None, q=0.0, max_iterations=5, silhouette_threshold=0.6
):
    if max_iterations < 1:
        raise ValueError("'max_iterations must be greater than zero")

    curr_data_est = np.zeros_like(data)

    line_lists = []
    silhouettes = []

    t_array, _ = gen_t_freq_arrays(N=len(data), dwell=dwell)

    n_peaks = 0

    silhouette_thresholds = np.linspace(silhouette_threshold, 0, max_iterations)

    for i in range(max_iterations):
        print(f'Iteration #{i}')
        curr_res = data - curr_data_est

        results = llc_kbdm(data=curr_res, dwell=dwell, m_range=m_range, p=p, l=l, q=q)

        if len(results.line_list) == 0:
            logging.info('No more peaks can be fitted. Stopping.')
            break

        filtered_index = np.nonzero(
            (results.silhouette > np.percentile(results.silhouette, silhouette_thresholds[i]))
        )

        line_list = results.line_list[filtered_index]

        curr_data_est_i = multi_fid(t_array=t_array, params=line_list)

        curr_data_est += curr_data_est_i

        line_lists.append(line_list)
        silhouettes.append(results.silhouette[filtered_index])

        n_peaks += len(line_list)

        print(f'Found {len(line_list)} peaks. Total: {n_peaks} peaks.')

    line_lists = np.r_[line_lists]
    line_list = np.concatenate(line_lists)
    silhouettes = np.r_[silhouettes]

    rmse = calculate_freq_domain_rmse(data=curr_data_est, params_est=line_list, dwell=dwell)

    return IterativeLlcKbdmResult(
        line_list=line_list,
        line_lists=line_lists,
        silhouettes=silhouettes,
        rmse=rmse
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


def _cluster_line_lists(samples, transformed_samples, method, method_parameters):
    """
    Cluster samples using non-supervised machine learning technique.

    :param numpy.ndarray samples:
        Array containing line lists

    :param numpy.ndarray samples:
        Array containing line lists represented in the transformed space

    :param str method:
        Clustering method. Can be 'hdbscan' or 'k-means'.

    :param dict method_parameters:
        Clustering method hyper-parameters.

    :return: number of estimated clusters, labels for each sample (with same dimensions of samples input),
        list containing labels of each cluster and labels of non-clustered samples.

    :rtype: tuple(int, numpy.ndarray, list(tuple), tuple)
    """
    if method == 'hdbscan':
        cl_model = hdbscan.HDBSCAN(**method_parameters)
    else:
        assert method == 'k-means', 'Invalid method'
        cl_model = KMeans(**method_parameters)

    cl_model.fit(transformed_samples)
    labels = cl_model.labels_

    num_clusters = len(set(labels) - {-1})

    clustered = []

    if num_clusters > 0:

        sample_silhouette_values = silhouette_samples(transformed_samples, labels)
        clustered_silhouettes = []

        for cluster_label in range(num_clusters):
            cluster = np.nonzero(labels == cluster_label)

            clustered.append(cluster)

            clustered_silhouettes.append(
                np.average(sample_silhouette_values[cluster])
            )

        non_clustered = np.nonzero(labels == -1)

        summarized_line_list = _summarize_clusters(
            samples=samples,
            clusters=clustered,
        )
    else:
        non_clustered = []
        summarized_line_list = []
        clustered_silhouettes = []

    return ClusteringResult(
        num_clusters=num_clusters,
        labels=labels,
        clustered=clustered,
        non_clustered=non_clustered,
        summarized_line_list=summarized_line_list,
        clustered_silhouettes=clustered_silhouettes,
    )


def _summarize_clusters(samples, clusters, summarizer=np.average):
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
        summarizer = np.average

    for cluster in clusters:
        cluster_samples = samples[cluster]
        cluster_samples[:, 1] = 1 / cluster_samples[:, 1]
        summarized_cluster = summarizer(cluster_samples, axis=0)
        summarized_cluster[1] = 1 / summarized_cluster[1]

        line_list.append(summarized_cluster)

    return np.array(line_list)


def filter_with_optics(samples, transformed_line_list):
    filter_model = OPTICS()

    # trying to find clusters structures on both ordinary and transformed space
    samples_filtered = filter_model.fit(samples).labels_ > 0
    transformed_samples_filtered = filter_model.fit(transformed_line_list).labels_ > 0

    transformed_line_list = transformed_line_list[samples_filtered | transformed_samples_filtered]
    samples = samples[samples_filtered | transformed_samples_filtered]

    assert len(transformed_line_list) == len(samples), \
        'Transformed and ordinary space should have the same output dimension after filtering.'

    return samples, transformed_line_list
