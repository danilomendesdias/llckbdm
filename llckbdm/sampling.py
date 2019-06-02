import logging

from llckbdm.kbdm import kbdm

logger = logging.getLogger(__name__)


def sample_kbdm(data, dwell, m_range, p, l, q=0, filter_invalid_features=True):
    """
    Compute samples of multiple computations of KBDM in a given range of m values.

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

    :param int|TypeNone l:
        This is used only with if gep_solver is set to 'svd'.
        ..see:: llckbdm.kbdm._solve_gep_svd
        Default is None.

    :param float q:
        This is used only with if gep_solver is set to 'svd'.
        ..see:: llckbdm.kbdm._solve_gep_svd
        Default is 0.

    :param bool filter_invalid_features:
        Apply filter_samples on each sample.
        Default is True.

    :return: Tuple containing a list with obtained features for each value of m and respective dictionary containing
    KBDM computations meta-information.
    :rtype: tuple(list(numpy.array), dict)
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
            l=l,
            q=q,
        )

        if filter_invalid_features:
            line_list = filter_samples(line_list)

        if len(line_list) > 0:
            logger.debug(f'Empty line list')
            line_lists.append(line_list)
            infos.append(info)

    return line_lists, infos


def filter_samples(samples, amplitude_tol=1e-6):
    """
    Filter samples by removing amplitudes below a certain threshold and negative transversal relaxation times.

    :param numpy.ndarray samples:
        Samples containing line lists to be filtered .

    :param float amplitude_tol:
        Cut-off value for amplitudes.
        Default is 1e-10.

    :return: filtered samples
    :rtype: numpy.ndarray
    """
    if len(samples) == 0:
        return samples

    amplitude_filter = samples[:, 0] > amplitude_tol
    T2_filter = samples[:, 1] > 0

    filtered_samples = samples[amplitude_filter & T2_filter]

    return filtered_samples
