import pytest
import numpy as np

from llckbdm.kbdm import kbdm
from llckbdm.llckbdm import _sample_kbdm, llc_kbdm, _transform_line_lists, _inverse_transform_line_lists, \
    _filter_samples, _cluster_line_lists


@pytest.fixture
def m_150_kbdm_line_list(data_brain_sim, dwell):
    line_list, _ = kbdm(
        data=data_brain_sim,
        dwell=dwell,
        m=150,
    )

    return line_list


def test_llc_sample_kbdm_should_raise_value_error_if_m_range_is_invalid(data_brain_sim, dwell):
    with pytest.raises(ValueError) as except_info:
        llc_kbdm(
            data=data_brain_sim,
            dwell=dwell,
            m_range=[1],
        )
    assert "size of 'm_range' must be greater than 2" in str(except_info.value)


def test_llc_sample_kbdm(data_brain_sim, dwell):
    m_range = range(100, 103, 1)

    line_lists, infos = _sample_kbdm(
        data=data_brain_sim,
        dwell=dwell,
        m_range=m_range,
        p=1,
        gep_solver='svd',
        l=None,
        q=None,
    )

    assert len(line_lists) == len(m_range)
    assert len(infos) == len(m_range)

    line_list_0_expected, info_expected = kbdm(
        data=data_brain_sim,
        dwell=dwell,
        m=m_range[0],
        gep_solver='svd',
        p=1,
    )

    assert line_lists[0] == pytest.approx(line_list_0_expected)
    assert infos[0] == pytest.approx(info_expected)


def test_transform_line_lists(params_brain_sim, dwell):
    transf_line_lists = _transform_line_lists(params_brain_sim, dwell)

    MU_RE = transf_line_lists[:, 0]
    MU_IM = transf_line_lists[:, 1]
    A = transf_line_lists[:, 2]
    PH = transf_line_lists[:, 3]

    MU = MU_RE + 1j * MU_IM

    OMEGA = np.log(MU) / (1j * dwell)

    F = np.real(OMEGA) / (2 * np.pi)
    T2 = 1. / np.imag(OMEGA)

    assert A == pytest.approx(params_brain_sim[:, 0])
    assert T2 == pytest.approx(params_brain_sim[:, 1])
    assert F == pytest.approx(params_brain_sim[:, 2])
    assert PH == pytest.approx(params_brain_sim[:, 3])


def test_inverse_transform_line_lists(params_brain_sim, dwell):
    transf_line_lists = _transform_line_lists(params_brain_sim, dwell)
    assert params_brain_sim == pytest.approx(_inverse_transform_line_lists(transf_line_lists, dwell))


def test_filter_samples_should_filter_invalid_kbdm_lines_for_data_without_noise(m_150_kbdm_line_list, params_brain_sim):

    filtered_samples = _filter_samples(m_150_kbdm_line_list)

    assert len(filtered_samples) == len(params_brain_sim)

    params_brain_sim = params_brain_sim[params_brain_sim[:, 0].argsort()]
    filtered_samples = filtered_samples[filtered_samples[:, 0].argsort()]

    assert filtered_samples[0] == pytest.approx(params_brain_sim[0], rel=0.01)
    assert filtered_samples[-1] == pytest.approx(params_brain_sim[-1], rel=0.01)


def test_filter_samples_should_not_crash_for_empty_input():
    empty_line_list = np.array([])

    assert np.array_equal(_filter_samples(empty_line_list), empty_line_list)


def test_cluster_line_lists_for_noiseless_data(data_brain_sim, dwell):
    m_range = range(150, 160, 1)
    l = 30
    k = 16  # number of genuine peaks

    line_lists, _ = _sample_kbdm(
        data=data_brain_sim,
        dwell=dwell,
        m_range=m_range,
        p=1,
        gep_solver='svd',
        l=l,
        q=0,
    )

    samples = np.concatenate(line_lists)

    num_clusters, labels, clustered, non_clustered = _cluster_line_lists(samples, eps=0.01, min_samples=len(m_range))

    assert num_clusters == len(clustered) == k

    non_clustered_samples = samples[non_clustered]

    assert np.shape(samples[non_clustered]) == (len(m_range) * (l - k), 4)

    assert _filter_samples(non_clustered_samples).size == 0

    for i in range(num_clusters):
        assert np.nonzero(labels == i) == pytest.approx(clustered[i], abs=0)
        assert np.shape(samples[clustered[i]]) == (len(m_range), 4)

    assert np.nonzero(labels == -1) == pytest.approx(non_clustered, abs=0)
