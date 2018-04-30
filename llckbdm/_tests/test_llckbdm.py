import pytest
import numpy as np

from llckbdm.sampling import filter_samples, sample_kbdm
from llckbdm.llckbdm import llc_kbdm, _transform_line_lists, _inverse_transform_line_lists, \
    _cluster_line_lists



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


def test_cluster_line_lists_for_noiseless_data(data_brain_sim, dwell):
    m_range = range(150, 160, 1)
    l = 30
    k = 16  # number of genuine peaks

    line_lists, _ = sample_kbdm(
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

    assert filter_samples(non_clustered_samples).size == 0

    for i in range(num_clusters):
        assert np.nonzero(labels == i) == pytest.approx(clustered[i], abs=0)
        assert np.shape(samples[clustered[i]]) == (len(m_range), 4)

    assert np.nonzero(labels == -1) == pytest.approx(non_clustered, abs=0)


def test_llc_kbdm(data_brain_sim, dwell, params_brain_sim, N):
    m_range = range(150, 250, 1)

    noise = np.random.randn(N,) + 1j * np.random.randn(N)
    noise = noise * 0.000

    line_list = llc_kbdm(
        data=data_brain_sim + noise,
        dwell=dwell,
        m_range=m_range,
        p=1,
        gep_solver='svd',
        l=30,
    )

    # assert len(line_list) == len(params_brain_sim)

    for i, param in enumerate(params_brain_sim):
        a_i, t2_i, f_i, ph_i = param[0], param[1], param[2], param[3]

        for j, est_param in enumerate(line_list):
            est_a_i, est_t2_i, est_f_i, est_ph_i = est_param[0], est_param[1], est_param[2], est_param[3]

            valid_a = est_a_i == pytest.approx(a_i, rel=0.3)
            valid_t2 = est_t2_i == pytest.approx(t2_i, rel=0.5)
            valid_f = est_f_i == pytest.approx(f_i, rel=0.03)
            valid_ph = est_ph_i == pytest.approx(ph_i, abs=0.01)

            if valid_a and valid_t2 and valid_f and valid_ph:
                break
            elif j + 1 == len(line_list):
                assert False, f'peak #{i} has failed in the test: {param}'


def test_llc_kbdm_should_raise_value_error_if_m_range_is_invalid(data_brain_sim, dwell):
    with pytest.raises(ValueError) as except_info:
        llc_kbdm(
            data=data_brain_sim,
            dwell=dwell,
            m_range=[1],
        )
    assert "size of 'm_range' must be greater than 2" in str(except_info.value)
