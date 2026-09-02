from __future__ import annotations

import copy

import numpy as np


def gating(
    src_signal,
    gate_peaks,
    gate_width=205,
    method=1,
):
    """Legacy gating function.

    Eliminate peaks (e.g. QRS) from src_signal using gates
    of width gate_width. The gate either filled by zeros or interpolation.
    The filling method for the gate is encoded as follows:
    0: Filled with zeros
    1: Interpolation samples before and after
    2: Fill with average of prior segment if exists
    otherwise fill with post segment
    3: Fill with running average of RMS (default)

    :param src_signal: Signal to process
    :type src_signalsignal: ~numpy.ndarray
    :param gate_peaks: list of individual peak index places to be gated
    :type gate_peaks: ~list
    :param gate_width: width of the gate
    :type gate_width: int
    :param method: filling method of gate
    :type method: int

    :returns: src_signal_gated, the gated result
    :rtype: ~numpy.ndarray
    """
    src_signal_gated = copy.deepcopy(src_signal)
    max_sample = src_signal_gated.shape[0]
    half_gate_width = gate_width // 2
    if method == 0:
        # Method 0: Fill with zeros
        # TODO: can rewrite with slices from numpy irange to be more efficient
        gate_samples = []
        for i, peak in enumerate(gate_peaks):
            for k in range(
                max(0, peak - half_gate_width),
                min(max_sample, peak + half_gate_width),
            ):
                gate_samples.append(k)

        src_signal_gated[gate_samples] = 0
    elif method == 1:
        # Method 1: Fill with interpolation pre- and post gate sample
        # TODO: rewrite with numpy interpolation for efficiency
        for i, peak in enumerate(gate_peaks):
            pre_ave_emg = src_signal[peak - half_gate_width - 1]

            if (peak + half_gate_width + 1) < src_signal_gated.shape[0]:
                post_ave_emg = src_signal[peak + half_gate_width + 1]
            else:
                post_ave_emg = 0

            k_start = max(0, peak - half_gate_width)
            k_end = min(peak + half_gate_width, src_signal_gated.shape[0])
            for k in range(k_start, k_end):
                frac = (k - peak + half_gate_width) / gate_width
                loup = (1 - frac) * pre_ave_emg + frac * post_ave_emg
                src_signal_gated[k] = loup

    elif method == 2:
        # Method 2: Fill with window length mean over prior section
        # ..._____|_______|_______|XXXXXXX|XXXXXXX|_____...
        #         ^               ^- gate start   ^- gate end
        #         - peak - half_gate_width * 3 (replacer)

        for i, peak in enumerate(gate_peaks):
            start = peak - half_gate_width * 3
            if start < 0:
                start = peak + half_gate_width
            end = start + gate_width
            pre_ave_emg = np.nanmean(src_signal[start:end])

            k_start = max(0, peak - half_gate_width)
            k_end = min(peak + half_gate_width, src_signal_gated.shape[0])
            for k in range(k_start, k_end):
                src_signal_gated[k] = pre_ave_emg

    elif method == 3:
        # Method 3: Fill with moving average over RMS
        gate_samples = []
        for i, peak in enumerate(gate_peaks):
            for k in range(
                max([0, int(peak - gate_width / 2)]),
                min([max_sample, int(peak + gate_width / 2)]),
            ):
                gate_samples.append(k)

        src_signal_gated_base = copy.deepcopy(src_signal_gated)
        src_signal_gated_base[gate_samples] = np.nan
        src_signal_gated_rms = full_rolling_rms(src_signal_gated_base, gate_width)

        for i, peak in enumerate(gate_peaks):
            k_start = max([0, int(peak - gate_width / 2)])
            k_end = min([int(peak + gate_width / 2), max_sample])

            for k in range(k_start, k_end):
                leftf = max([0, int(k - 1.5 * gate_width)])
                rightf = min([int(k + 1.5 * gate_width), max_sample])
                src_signal_gated[k] = np.nanmean(src_signal_gated_rms[leftf:rightf])

    return src_signal_gated


def full_rolling_rms(data_emg, window_length):
    """Leagcy rolling RMS calculation.

    This function computes a root mean squared envelope over an
    array :code:`data_emg`.  To do this it uses number of sample values
    :code:`window_length`. It differs from :func:`naive_rolling_rms` by that the
    output is the same length as the input vector.

    :param data_emg: Samples from the EMG
    :type data_emg: ~numpy.ndarray
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns: The root-mean-squared EMG sample data
    :rtype: ~numpy.ndarray
    """
    x_pad = np.pad(data_emg, (0, window_length - 1), "constant", constant_values=(0, 0))
    x_2 = np.power(x_pad, 2)
    window = np.ones(window_length) / float(window_length)
    emg_rms = np.sqrt(np.convolve(x_2, window, "valid"))
    return emg_rms
