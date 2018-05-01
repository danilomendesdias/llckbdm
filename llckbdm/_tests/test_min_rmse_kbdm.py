import pytest

from llckbdm.min_rmse_kbdm import min_rmse_kbdm


def test_min_rmse_kbdm(data_brain_sim, dwell):
    # because the number of points used to compute KBDM, only third element is capable of reproduce a good result
    m_range = [30, 31, 180, 32, 33, 34]
    l = 30

    min_rmse_results = min_rmse_kbdm(
        data=data_brain_sim,
        dwell=dwell,
        m_range=m_range,
        l=l
    )

    assert len(min_rmse_results.samples) == len(m_range)
    # it should compute rmse 0 for noiseless
    assert min_rmse_results.min_rmse == pytest.approx(0)

    # assert that third element produced the best result
    assert min_rmse_results.min_index == 2
