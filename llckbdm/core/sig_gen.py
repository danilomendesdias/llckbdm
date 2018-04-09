import numpy as np

# @TODO: add/fix docs

def fft(data):
    N = len(data)
    return np.fft.fftshift(np.fft.fft(data)) / np.sqrt(N)


def gen_t_freq(N, dwell):
    t = np.arange(0, N * dwell, dwell)
    freq = np.fft.fftshift(np.fft.fftfreq(N, dwell))
    
    return t, freq


def expdec(t, a, T2, force_abs=False):
    """Return exponential decay
    t: time axis
    a: amplitude
    force_abs: echo like
    """
    # FID like
    if not force_abs:
        return a * np.exp(-t / T2)
    # Echo like
    else:
        return a * np.exp(-np.abs(t) / T2)


def fid(t, a, T2, f, phase=0):
    """Return complex free induction decay.
    t: time axis
    a: echo amplitude
    T2: transversal relaxation time
    f: frequency
    phase: phase between imaginary and real components of echo
    """
    if T2 < 0.:
        return np.zeros(t.size, dtype=np.complex)

    data = np.cos(2 * np.pi * f * t + phase) + 1j * \
        np.sin(2 * np.pi * f * t + phase)
    return expdec(t, a, T2) * data


def echo(t, a, T2, f, phase=0):
    """Return complex echo.
    t: time axis
    a: echo amplitude
    T2: transversal relaxation time
    f: frequency
    phase: phase between imaginary and real components of echo
    """
    data = np.cos(2 * np.pi * f * t + phase) + 1j * \
        np.sin(2 * np.pi * f * t + phase)
    return expdec(t, a, T2, True) * data


def complex_exp(t, a, omega):
    return a * np.exp(-1j * t * omega)


def spec(freq, params):
    # signal at frequency domain
    sf = np.zeros(freq.size, dtype=np.complex)

    for param in params:
        # adding peak into spectrum
        sf = sf + peak(freq, *param)

    return sf


def peak(freq, a, T2, f, phase):
    A = a * np.exp(1j * phase)
    B = (1. / T2) + 2j * np.pi * (freq - f)

    dataf = A / B

    return dataf


def multi_fid(t, params):
    """
    Return a signal with multiple components
    t: time axis
    params: list with components parameters as follows:
        [[a_1, T2_1, f_1, phase_1],
         [a_2, T2_2, f_2, phase_2],
         ...
        ]
    """
    # signal
    s = np.zeros(t.size, dtype=np.complex)

    for param in params:
        # adding component into signal
        s = s + fid(t, *param)

    return s
