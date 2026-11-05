"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to visualize the power spectrum of EMG arrays.
"""  # noqa: D205

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram, welch


def show_power_spectrum(
    signal: np.ndarray,
    fs_emg: int,
    t_window_s: int,
    axis_spec: int = 1,
    signal_unit: str = "uV",
) -> tuple[np.ndarray, np.ndarray]:
    """Plot the power spectrum of the provided EMG signal.

    Plot the power spectrum of the frequencies comtained in an EMG based on
    a Fourier transform.  It does not return the graph, rather the values but
    plots the graph before it return.  Sample should be one single row
    (1-dimensional array).

    Parameters
    ----------
    signal : numpy.ndarray
        The signal array
    fs_emg : int
        emg sampling rate
    t_window_s : int
        The end of window over which values will be plotted
    axis_spec : int
        1 for logaritmic axis, 0 for linear axis (Default value = 1)
    signal_unit : str
        Unit of y-axis, default is uV

    Returns
    -------
    (yf, xf): (numpy.ndarray, numpy.ndarray)
        Fourier transformed array and frequencies axis

    Raises
    ------
    ValueError
        If the axis_spec value is invalid.

    """
    n_samples = len(signal)
    # for our emgs sampling rate is usually 2048
    y_f = np.abs(fft(signal)) ** 2
    x_f = fftfreq(n_samples, 1 / fs_emg)

    idx = [i for i, v in enumerate(x_f) if 0 <= v <= t_window_s]
    psd_label = f"PSD [{signal_unit}**2/Hz]"

    if axis_spec == 1:
        plt.semilogy(x_f[idx], y_f[idx])
    elif axis_spec == 0:
        plt.plot(x_f[idx], y_f[idx])
    else:
        msg = (
            "Invalid axis_spec value. Please use 1 "
            "for logarithmic axis or 0 for linear axis."
        )
        raise ValueError(msg)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(psd_label)
    plt.title("Power Spectral Density")
    plt.show()

    return y_f, x_f


def show_psd_welch(
    signal: np.ndarray,
    fs_emg: int,
    t_window_s: int,
    axis_spec: int = 1,
    signal_unit: str = "uV",
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the power spectrum density using the Welch method.

    This method involves dividing the signal into overlapping segments,
    copmuting a modified periodogram for each segment, and then averaging these
    periodograms.

    Parameters
    ----------
    signal : numpy.ndarray
        the signal array
    fs_emg : int
        Number of samples per second
    t_window_s : int
        Length of segments in which  original signal is divided
    axis_spec : int : {0, 1}, default 1
        1 for logaritmic axis, 0 for linear axis (Default value = 1)
    signal_unit : str

    Returns
    -------
    (f, Pxx_den): tuple of numpy.ndarray
        Frequencies and corresponding power spectral density values

    Raises
    ------
    ValueError
        - If the signal is not 1-dimensional
        - If the axis_spec value is invalid

    """
    if signal.ndim != 1:
        msg = "Sample array must be 1-dimensional"
        raise ValueError(msg)

    window = np.hanning(t_window_s)
    f, pxx_den = welch(signal, fs_emg, window=window, nperseg=t_window_s)
    psd_label = f"PSD [{signal_unit}**2/Hz]"

    if axis_spec == 1:
        plt.semilogy(f, pxx_den)
    elif axis_spec == 0:
        plt.plot(f, pxx_den)
    else:
        msg = (
            "Invalid axis_spec value. Please use 1 "
            "for logarithmic axis or 0 for linear axis."
        )
        raise ValueError(msg)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(psd_label)
    plt.title("Power Spectral Density")
    plt.show()

    return f, pxx_den


def show_periodogram(
    signal: np.ndarray,
    fs_emg: int,
    axis_spec: int = 1,
    signal_unit: str = "uV",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and display the periodogram of a signal.

    This function calculates and shows the periodogram of the provided EMG
    signal. The resulting plot can be displayed with either a logarithmic or
    linear y-axis, based on the specified axis_spec parameter.


    Parameters
    ----------
    signal : numpy.ndarray
        the signal array
    fs_emg : int
        emg sampling rate
    axis_spec : int, default=1
        1 for logaritmic axis, 0 for linear axis
    signal_unit : str, default='uV'
        Unit of y-axis

    Returns
    -------
    (f, Pxx_den): tuple of numpy.ndarray
        Frequencies and corresponding power spectral density values

    Raises
    ------
    ValueError
        If the axis_spec value is invalid.
    """
    f, Pxx_den = periodogram(signal, fs_emg)  # noqa: N806
    psd_label = f"PSD [{signal_unit}**2/Hz]"

    if axis_spec == 1:
        plt.semilogy(f, Pxx_den)
    elif axis_spec == 0:
        plt.plot(f, Pxx_den)
    else:
        msg = (
            "Invalid axis_spec value. Please use 1 for "
            "logarithmic axis or 0 for linear axis."
        )
        raise ValueError(msg)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(psd_label)
    plt.title("Periodogram")
    plt.show()

    return f, Pxx_den
