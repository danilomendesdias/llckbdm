import pytest
import numpy as np

from llckbdm.kbdm import kbdm
from llckbdm.sampling import sample_kbdm, filter_samples


@pytest.fixture
def m_150_kbdm_line_list(data_brain_sim, dwell):
    line_list, _ = kbdm(
        data=data_brain_sim,
        dwell=dwell,
        m=150,
    )

    return line_list


def test_sample_kbdm(data_brain_sim, dwell):
    m_range = range(100, 103, 1)

    line_lists, infos = sample_kbdm(
        data=data_brain_sim,
        dwell=dwell,
        m_range=m_range,
        p=1,
        l=None,
        q=0,
        filter_invalid_features=False
    )

    assert len(line_lists) == len(m_range)
    assert len(infos) == len(m_range)

    line_list_0_expected, info_expected = kbdm(
        data=data_brain_sim,
        dwell=dwell,
        m=m_range[0],
        p=1,
        l=None
    )

    assert line_lists[0] == pytest.approx(line_list_0_expected)

    assert infos[0].m == info_expected.m
    assert infos[0].l == info_expected.l
    assert infos[0].p == info_expected.p
    assert infos[0].q == info_expected.q
    assert infos[0].singular_values == pytest.approx(info_expected.singular_values)


def test_filter_samples_should_filter_invalid_kbdm_lines_for_data_without_noise(m_150_kbdm_line_list, params_brain_sim):

    filtered_samples = filter_samples(m_150_kbdm_line_list)

    assert len(filtered_samples) == len(params_brain_sim)

    params_brain_sim = params_brain_sim[params_brain_sim[:, 0].argsort()]
    filtered_samples = filtered_samples[filtered_samples[:, 0].argsort()]

    assert filtered_samples[0] == pytest.approx(params_brain_sim[0], abs=0.01)
    assert filtered_samples[-1] == pytest.approx(params_brain_sim[-1], abs=0.01)


def test_filter_samples_should_not_crash_for_empty_input():
    empty_line_list = np.array([])

    assert np.array_equal(filter_samples(empty_line_list), empty_line_list)
