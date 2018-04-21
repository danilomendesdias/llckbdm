import pytest
import numpy as np

from llckbdm.kbdm import kbdm
from llckbdm.llckbdm import _sample_kbdm, llc_kbdm, _transform_line_lists, _inverse_transform_line_lists


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
