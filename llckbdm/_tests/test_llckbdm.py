import sys

import pytest
import numpy as np
from llckbdm.sig_gen import multi_fid

from llckbdm.sampling import filter_samples
from llckbdm.llckbdm import llc_kbdm, _transform_line_lists, _inverse_transform_line_lists, iterative_llc_kbdm


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


def test_llc_kbdm(t_array, data_brain_sim, dwell, params_brain_sim, N):
    m_range = range(250, 260, 1)

    results = llc_kbdm(
        data=data_brain_sim,
        dwell=dwell,
        m_range=m_range,
        p=1,
        l=30,
    )

    line_list = results.line_list

    line_list = filter_samples(line_list, amplitude_tol=1e-3)

    assert len(line_list) == len(params_brain_sim)

    est_data = multi_fid(t_array, line_list)

    assert np.std(est_data.real - data_brain_sim.real) < 1e-3
    assert np.std(est_data.imag - data_brain_sim.imag) < 1e-3


def test_llc_kbdm_should_raise_value_error_if_m_range_is_invalid(data_brain_sim, dwell):
    with pytest.raises(ValueError) as except_info:
        llc_kbdm(
            data=data_brain_sim,
            dwell=dwell,
            m_range=[1],
        )
    assert "size of 'm_range' must be greater than 2" in str(except_info.value)


@pytest.mark.xfail(
    condition=sys.platform.startswith('darwin'),
    reason='Reported in https://github.com/danilomendesdias/llckbdm/issues/19'
)
def test_iterative_llc_kbdm(data_brain_sim, dwell):
    iterative_llc_kbdm(data=data_brain_sim, dwell=dwell, m_range=range(180, 190))
