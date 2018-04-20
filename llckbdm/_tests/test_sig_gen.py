import numpy as np
import pytest

from llckbdm.sig_gen import fid, multi_fid, gen_t_freq_arrays, lorentzian_peak, spec, _validate_parameters


@pytest.fixture
def params():
    a_1, t2_1, f_1, phase_1 = 1., 0.1, 0, 0.
    a_2, t2_2, f_2, phase_2 = 2., 0.01, 100, np.pi / 2.

    params_ = [
        [a_1, t2_1, f_1, phase_1],
        [a_2, t2_2, f_2, phase_2]
    ]

    return params_


def test_gen_t_freq_arrays(N, dwell):
    t_array, freq_array = gen_t_freq_arrays(N, dwell)

    assert len(t_array) == N
    assert len(freq_array) == N

    assert t_array[0] == pytest.approx(0)
    assert t_array[-1] == pytest.approx((N - 1) * dwell)

    assert freq_array[0] == pytest.approx(- 1 / (2 * dwell))
    assert freq_array[-1] == pytest.approx((0.5 - 1/N) / dwell)


def test_fid(t_array):
    a, t2, f, phase = 1., 1., 1., 0.

    data = fid(
        t_array=t_array,
        a=a,
        t2=t2,
        f=f,
        phase=phase
    )

    assert data[0].real == pytest.approx(a)
    assert data[0].imag == pytest.approx(0)

    assert np.abs(data) == pytest.approx(a * np.exp(-t_array / t2))

    assert np.angle(data[0]) == pytest.approx(phase)

    assert data[100] == pytest.approx(
        a * np.exp(-t_array[100] / t2) * np.exp(1j * (2 * np.pi * f * t_array[100] + phase))
    )

    assert data[-1] == pytest.approx(
        a * np.exp(-t_array[-1] / t2) * np.exp(1j * (2 * np.pi * f * t_array[-1] + phase))
    )


def test_multi_fid(t_array, params):

    data = multi_fid(
        t_array=t_array,
        params=params
    )

    fid_0 = fid(t_array, *params[0])
    fid_1 = fid(t_array, *params[1])

    assert fid_0 + fid_1 == pytest.approx(data)


def test_lorentzian_peak(freq_array):
    a, t2, f, phase = 3.5, 0.05, 350., 0.

    data_peak = lorentzian_peak(
        freq_array=freq_array,
        a=a,
        t2=t2,
        f=f,
        phase=phase
    )

    assert freq_array[np.argmax(data_peak.real)] == pytest.approx(f, abs=1)
    assert np.trapz(data_peak.real, freq_array) == pytest.approx(a / 2, abs=0.1)


def test_spec(freq_array, params):
    data_spec = spec(
        freq_array=freq_array,
        params=params
    )

    peak_0 = lorentzian_peak(freq_array, *params[0])
    peak_1 = lorentzian_peak(freq_array, *params[1])

    assert peak_0 + peak_1 == pytest.approx(data_spec)


def test_validate_parameters_should_raise_value_error_for_invalid_params():
    with pytest.raises(ValueError) as except_info:
        _validate_parameters(
            a=-1,
            t2=2,
            f=1,
            phase=1,
        )

    assert "Amplitude can't be negative" in str(except_info.value)

    with pytest.raises(ValueError) as except_info:
        _validate_parameters(
            a=1,
            t2=-2,
            f=1,
            phase=1,
        )

    assert "T2 must be positive" in str(except_info.value)


def test_validate_parameters_should_display_warning_for_phase_greater_than_2_pi(caplog):
    _validate_parameters(
        a=1,
        t2=2,
        f=1,
        phase=10,
    )

    assert 'Check whether the correct unit is being used' in caplog.text
