import logging

import numpy as np

logger = logging.getLogger(__name__)


def gen_t_freq_arrays(N, dwell):
    """
    Generate time and frequencies array for a given number of points and dwell time.

    :param int N:
        Number of points.

    :param float dwell:
        Dwell time in seconds.

    :return: Time array and shifted array of frequencies.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    t_array = np.arange(0, N * dwell, dwell)
    freq_array = np.fft.fftshift(np.fft.fftfreq(N, dwell))

    return t_array, freq_array


def fid(t_array, a, t2, f, phase=0.):
    """
    Calculate Free Induction Decay (FID) for a given time array.

    :param  t_array:
        ..see:: multi_fid

    :param a:
        ..see:: _validate_parameters

    :param T2:
        ..see:: _validate_parameters

    :param f:
        ..see:: _validate_parameters

    :param phase:
        ..see:: _validate_parameters.
        Default is 0.

    :return: Array containing FID
    :rtype: numpy.ndarray
    """
    _validate_parameters(a, t2, f, phase)

    data_harmonic = np.exp(1j * (2 * np.pi * f * t_array + phase))

    return a * np.exp(-t_array / t2) * data_harmonic


def multi_fid(t_array, params):
    """
    Compute time domain signal composed by multiple Free Induction Decay signals.

    :param numpy.ndarray t_array:
        Array of time in seconds.

    :param list(tuple(float, float, float, float)) params:
        List containing parameters of each peak.
        Each parameter must be given in the following order: (amplitude, t2, frequency, phase,).

    :return: Array containing time-domain signal with multiple FIDs
    :rtype: numpy.ndarray
    """
    return np.sum([fid(t_array, *param) for param in params], axis=0)


def fft(data):
    """
    Compute normalized shifted Fourier Transform of input signal.

    ..see:: numpy.fft.fft
    ..see:: numpy.fft.fftshift

    :param numpy.ndarray data:
        Numpy array with complex data points.

    :return: Array containing normalized shifted Fourier Transform of data
    :rtype: numpy.ndarray
    """
    N = len(data)
    return np.fft.fftshift(np.fft.fft(data)) / np.sqrt(N)


def lorentzian_peak(freq_array, a, t2, f, phase=0):
    """
    Calculate a lorentzian peak for a given frequency array.

    :param freq_array:
        ..see:: spec

    :param a:
        ..see:: _validate_parameters

    :param t2:
        ..see:: _validate_parameters

    :param f:
        ..see:: _validate_parameters

    :param phase:
        ..see:: _validate_parameters. 
        Default is 0.

    :return: Array containing a lorentzian peak in the frequency domain.
    :rtype: numpy.ndarray
    """
    _validate_parameters(a, t2, f, phase)

    A = a * np.exp(1j * phase)
    B = (1. / t2) + 2j * np.pi * (freq_array - f)

    data_peak = A / B

    return data_peak


def spec(freq_array, params):
    """
    Compute spectrum for a given array of frequencies and peak parameters.

    :param numpy.ndarray freq_array:
        Array of frequencies in Hz.

    :param params:
        ..see:: multi_fid

    :return: Array containing multiple lorentzian peaks in the frequency domain
    :rtype: numpy.ndarray
    """
    return np.sum([lorentzian_peak(freq_array, *param) for param in params], axis=0)


def _validate_parameters(a, t2, f, phase):
    """
    Validate parameter of a given component.
    Can raise ValueError on critical cases or display warnings.

    @TODO: include dwell time and number of points as arguments and use them to validate frequency.

    :param float a:
        Amplitude (arbitrary units). Can't be negative.

    :param float t2:
        Transversal relaxation time (1/s). Must be positive.

    :param float f:
        Frequency (Hz).

    :param float phase:
        Phase (rad/s). Shows warning if greater than 2 * pi.
    """
    if t2 <= 0:
        raise ValueError("T2 must be positive.")

    if a < 0:
        raise ValueError("Amplitude can't be negative.")

    if np.abs(phase) > 2 * np.pi:
        logger.warning(
            'Phase is greater than 2 * pi and phase must be given in rad/s. '
            'Check whether the correct unit is being used.'
        )
