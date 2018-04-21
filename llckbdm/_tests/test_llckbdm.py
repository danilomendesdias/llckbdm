import pytest

from kbdm import kbdm
from llckbdm.llckbdm import _sample_kbdm


def test_llc_sample_kbdm(data_brain_sim, dwell):
    m_range = range(100, 103, 1)

    line_lists = _sample_kbdm(
        data=data_brain_sim,
        dwell=dwell,
        m_range=m_range,
        p=1,
        gep_solver='svd',
        l=None,
        q=None,
    )

    assert len(line_lists) == len(m_range)

    line_list_0_expected = kbdm(
        data=data_brain_sim,
        dwell=dwell,
        m=m_range[0],
        gep_solver='svd',
        p=1,
    )

    assert line_lists[0] == pytest.approx(line_list_0_expected)
