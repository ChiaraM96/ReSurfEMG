"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to eliminate ECG artifacts from EMG arrays.
"""  # noqa: D205

import numpy as np
import pandas as pd
import pywt
import scipy

import resurfemg.preprocessing.filtering as filt
from resurfemg.preprocessing import envelope as evl

# Gate fill method constants (avoid magic-number comparisons)
GATE_FILL_ZEROS = 0  # 0: filled with zeros
GATE_FILL_INTERP = 1  # 1: interpolation
GATE_FILL_PRIOR_MEAN = 2  # 2: prior-window mean
GATE_FILL_RMS = 3  # 3: RMS running average


def detect_ecg_peaks(
    ecg_raw: np.ndarray,
    fs: int,
    peak_fraction: float = 0.4,
    peak_width_s: int | None = None,
    peak_distance: int | None = None,
    bp_filter: bool = True,
) -> np.ndarray:
    """Detect ECG peaks in EMG signal.

    Parameters
    ----------
    ecg_raw : numpy.ndarray
        ecg or emg signal to detect the ECG peaks in.
    fs : int
        Sampling rate of the emg signals.
    peak_fraction : float, default=0.4
        ECG peaks amplitude threshold relative to the
        specified fraction of the min-max values in the ECG signal
    peak_width_s : int, optional
        ECG peaks width threshold in samples.
    peak_distance : int, optional
        Minimum time between ECG peaks in samples.
    bp_filter : bool, default=True
        If `True`, bandpass filter the ecg_raw between 1-500 Hz before peak
        detection.

    Returns
    -------
    ecg_peak_idxs: numpy.ndarray[int]
        ecg_peak_idxs: ECG peak indices

    """
    if peak_width_s is None:
        peak_width_s = fs // 1000

    if peak_distance is None:
        peak_distance = fs // 3

    if bp_filter:
        lp_cf = min([500, 0.95 * fs / 2])
        ecg_filt = filt.emg_bandpass_butter(
            ecg_raw, high_pass=1, low_pass=lp_cf, fs_emg=fs
        )
        ecg_rms = evl.full_rolling_rms(ecg_filt, fs // 200)
    else:
        ecg_rms = evl.full_rolling_rms(ecg_raw, fs // 200)
    max_ecg_rms = np.percentile(ecg_rms, 99)
    min_ecg_rms = np.percentile(ecg_rms, 1)
    peak_height = peak_fraction * (max_ecg_rms - min_ecg_rms)

    ecg_peak_idxs, _ = scipy.signal.find_peaks(
        ecg_rms, height=peak_height, width=peak_width_s, distance=peak_distance
    )

    return ecg_peak_idxs


def gating(  # noqa: C901, PLR0915
    emg_raw: np.ndarray,
    peak_idxs: np.ndarray,
    gate_width: int = 205,
    method: int = GATE_FILL_RMS,
) -> np.ndarray:
    """Remove ECG peaks from EMG signal using gating.

    Eliminate peaks (e.g. QRS) from emg_raw using gates of width `gate_width`.
    The gates are centered on the indices in `peak_idxs`. The gated samples are
    filled using one of the following methods:
    - 0: Filled with zeros
    - 1: Interpolation samples before and after
    - 2: Fill with average of prior segment if exists, post segment otherwise.
    - 3: Fill with running average of RMS (default).

    Parameters
    ----------
    emg_raw : numpy.ndarray
        Signal to process
    peak_idxs : numpy.ndarray[int]
        list of individual peak index places to be gated
    gate_width : int, default=205
        width of the gate
    method : int, default=3
        filling method of gate

    Returns
    -------
    emg_raw_gated: numpy.ndarray
        emg_raw_gated: the emg signal with ECG peaks removed and filled
            according to the specified method.

    """

    def _masked_window_means(
        source: np.ndarray,
        start_idxs: np.ndarray,
        width: int,
    ) -> np.ndarray:
        """Windowed masked mean.

        Calculate means over [start, start + width) windows of `source`,
        ignoring both out-of-bounds positions and NaN values via a MaskedArray.
        Fully-empty windows return NaN.

        Parameters
        ----------
        source : numpy.ndarray
            1D array to average over.
        start_idxs : numpy.ndarray
            1D array of window start indices.
        width : int
            Window length.


        Returns
        -------
        numpy.ndarray
            Per-window mean, NaN where the window is fully masked.

        """
        # Create a 2D array of indices for each window, shape (peaks, window)
        idx_mat = start_idxs[:, None] + np.arange(width)
        # Out-of-bounds mask
        oob = (idx_mat < 0) | (idx_mat >= source.shape[0])
        # Gather values, clip to bounds, and mask out-of-bounds and NaN values
        vals = source[np.clip(idx_mat, 0, source.shape[0] - 1)]
        # Mask out-of-bounds and NaN values, then compute mean
        masked = np.ma.array(vals, mask=oob | np.isnan(vals))
        # Return the mean of each window, with NaN for fully masked windows
        return masked.mean(axis=1).filled(np.nan)

    def _get_emg_raw_gated_rms(
        emg_raw_gated: np.ndarray, gate_mask: np.ndarray, gate_width: int
    ) -> np.ndarray:
        """Calculate the RMS of the gated EMG signal.

        Parameters
        ----------
        emg_raw_gated : numpy.ndarray
            The gated EMG signal.
        gate_mask : numpy.ndarray
            A boolean mask indicating the locations of the gates.
        gate_width : int
            The width of the gates.


        Returns
        -------
        numpy.ndarray
            The RMS of the gated EMG signal.

        """
        emg_raw_gated_base = np.array(emg_raw_gated, copy=True)
        emg_raw_gated_base[gate_mask] = np.nan
        return evl.full_rolling_rms(emg_raw_gated_base, gate_width)

    def _find_corresponding_peak(
        peak_idxs: np.ndarray,
        gate_idxs: np.ndarray,
        gate_width: int,
        max_sample: int,
    ) -> np.ndarray:
        """Find the corresponding peak index for each gated sample index.

        Parameters
        ----------
        peak_idxs : numpy.ndarray
            Indices of the detected peaks.
        gate_idxs : numpy.ndarray
            Indices of the gated samples.
        gate_width : int
            The width of the gate.
        max_sample : int.
            The maximum sample index (length of the EMG signal).


        Returns
        -------
        numpy.ndarray
            The corresponding peak indices for each gated sample index.

        """
        start_idxs = np.clip(
            peak_idxs - gate_width // 2 + 1, 0, max_sample
        ).astype(int)
        matched_peak = np.searchsorted(start_idxs, gate_idxs, side="right") - 1
        return np.clip(matched_peak, 0, len(peak_idxs) - 1).astype(int)

    def _interp_missing_samples(
        emg_raw_gated: np.ndarray,
        interp_mask: np.ndarray,
        sample_idxs: np.ndarray,
    ) -> np.ndarray:
        """Interpolate missing samples in the gated EMG signal.

        Parameters
        ----------
        emg_raw_gated : numpy.ndarray
            The gated EMG signal.
        interp_mask : numpy.ndarray
            A boolean mask indicating the locations of the samples to
            interpolate.
        sample_idxs : numpy.ndarray
            The indices of the samples in the EMG signal.

        Returns
        -------
        numpy.ndarray
            The EMG signal with the missing samples interpolated.

        """
        interpolate_samples = sample_idxs[interp_mask]
        if interpolate_samples.size == 0:
            return emg_raw_gated

        x_samp = np.arange(emg_raw_gated.shape[0])
        other_mask = ~np.isin(x_samp, interpolate_samples)
        if not other_mask.any():  # no anchors left to interpolate from
            return emg_raw_gated

        emg_raw_gated[interpolate_samples] = np.interp(
            interpolate_samples, x_samp[other_mask], emg_raw_gated[other_mask]
        )
        return emg_raw_gated

    def _gate_fill_interpolation(
        fill_idxs: np.ndarray,
        emg_raw_gated: np.ndarray,
        gated_idxs: np.ndarray,
        peak_idxs: np.ndarray,
        gate_width: int,
    ) -> np.ndarray:
        """Fill gates using linear interpolation.

        Fill the gated EMG signal using linear interpolation between the values
        before and after the gate.

        Parameters
        ----------
        emg_raw_gated : numpy.ndarray
            The gated EMG signal.
        gated_idxs : numpy.ndarray
            Indices of the gated samples.
        peak_idxs : numpy.ndarray
            Indices of the detected peaks.
        gate_width : int
            The width of the gate.
        fill_idxs : np.ndarray


        Returns
        -------
        numpy.ndarray
            The filled EMG signal.

        """
        if gated_idxs.size == 0:
            return emg_raw_gated

        # Compute the indices of the samples just before and after each gate
        # window
        pre_idxs = -gate_width // 2 + peak_idxs
        post_idxs = gate_width // 2 + peak_idxs + 2
        # Check bounds
        pre_in = (pre_idxs >= 0) & (pre_idxs < emg_raw_gated.shape[0])
        post_in = (post_idxs >= 0) & (post_idxs < emg_raw_gated.shape[0])
        # If the indices are out of bounds, use 0.0 as the value
        pre_val = np.where(
            pre_in,
            emg_raw_gated[np.clip(pre_idxs, 0, emg_raw_gated.shape[0] - 1)],
            0.0,
        )
        post_val = np.where(
            post_in,
            emg_raw_gated[np.clip(post_idxs, 0, emg_raw_gated.shape[0] - 1)],
            0.0,
        )

        # Find the corresponding peak for each gated index
        peak_nrs = _find_corresponding_peak(
            peak_idxs, gated_idxs, gate_width, emg_raw_gated.shape[0]
        )
        # Calculate the fraction of the gate width for each gated index
        frac = (gated_idxs - peak_idxs[fill_idxs] + gate_width // 2) / float(
            gate_width
        )
        # Fill the gated EMG signal using linear interpolation
        emg_raw_gated[gated_idxs] = (1.0 - frac) * pre_val[
            peak_nrs
        ] + frac * post_val[peak_nrs]
        return emg_raw_gated

    def _gate_fill_prior_mean(
        emg_raw: np.ndarray,
        emg_raw_gated: np.ndarray,
        gated_idxs: np.ndarray,
        peak_idxs: np.ndarray,
        gate_width: int,
    ) -> np.ndarray:
        """Fill the gated EMG signal with the mean of a window before the gate.

        Parameters
        ----------
        emg_raw : numpy.ndarray
            The original EMG signal.
        emg_raw_gated : numpy.ndarray
            The gated EMG signal.
        gated_idxs : numpy.ndarray
            Indices of the gated samples.
        peak_idxs : numpy.ndarray
            Indices of the detected peaks.
        gate_width : int
            The width of the gate.


        Returns
        -------
        numpy.ndarray
            The filled EMG signal.

        """
        prior_start_idxs = peak_idxs - (gate_width // 2) * 3
        oob = prior_start_idxs < 0
        prior_means = _masked_window_means(
            emg_raw, prior_start_idxs, gate_width
        )
        post_means = _masked_window_means(
            emg_raw, peak_idxs + gate_width // 2, gate_width
        )

        fill_means = np.where(oob, post_means, prior_means)
        # Find the corresponding peak for each gated index
        peak_nrs = _find_corresponding_peak(
            peak_idxs, gated_idxs, gate_width, emg_raw_gated.shape[0]
        )
        emg_raw_gated[gated_idxs] = fill_means[peak_nrs]
        return emg_raw_gated

    def _gate_fill_rms(
        emg_raw_gated: np.ndarray,
        gate_mask: np.ndarray,
        gated_idxs: np.ndarray,
        gate_width: int,
    ) -> np.ndarray:
        """Fill the gates by interpolating the RMS of the surrounding signal.

        Parameters
        ----------
        emg_raw_gated : numpy.ndarray
            The gated EMG signal.
        gate_mask : numpy.ndarray
            A boolean mask of the gate locations.
        gated_idxs : numpy.ndarray
            Indices of the gated samples.
        gate_width : int
            The width of the gate.


        Returns
        -------
        numpy.ndarray
            The filled EMG signal.

        """
        emg_raw_gated_rms = _get_emg_raw_gated_rms(
            emg_raw_gated, gate_mask, gate_width
        )

        w = max(1, 2 * int(1.5 * gate_width) + 1)
        closed = "neither" if gate_width % 2 == 0 else "left"
        local_rms = (
            pd.Series(emg_raw_gated_rms)
            .rolling(window=w, min_periods=1, center=True, closed=closed)
            .mean()
            .values
        )

        fill_vals = np.asarray(local_rms[gated_idxs])
        interp_mask = np.isnan(fill_vals)
        emg_raw_gated[gated_idxs[~interp_mask]] = fill_vals[~interp_mask]

        return _interp_missing_samples(
            emg_raw_gated,
            interp_mask,
            gated_idxs,
        )

    def _compute_gated_indices(
        peak_idxs: np.ndarray,
        gate_width: int,
        max_sample: int,
    ) -> np.ndarray:
        """Compute the indices of the gated samples.

        Parameters
        ----------
        peak_idxs : np.ndarray
            The indices of the ECG peaks to gate.
        gate_width : int
            The width of the gates.
        max_sample : int
            The maximum sample index (length of the EMG signal).


        Returns
        -------
        numpy.ndarray
            The indices of the gated samples, clipped to the valid range of
            the EMG signal.

        """
        gated_idxs = (
            peak_idxs[:, None]
            + np.arange(-gate_width // 2, gate_width // 2)
            + 1
        ).ravel()

        return gated_idxs[
            (gated_idxs >= 0) & (gated_idxs < max_sample)
        ].astype(int)

    def _compute_gate_mask(
        peak_idxs: np.ndarray,
        gate_width: int,
        max_sample: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a boolean mask for the gated samples.

        Odd gate widths are centered on the peak, while even gate widths have
        the peak at the left of center.

        Overlapping windows are merged using a difference-array and cumulative
        sum approach.

        Parameters
        ----------
        peak_idxs : np.ndarray
            The indices of the ECG peaks to gate.
        gate_width : int
            The width of the gates.
        max_sample : int
            The maximum sample index (length of the EMG signal).


        Returns
        -------
        np.ndarray
            A boolean mask indicating the locations of the gated samples.
        np.ndarray
            The indices of the gated samples.

        """
        k_start = np.clip(
            -gate_width // 2 + peak_idxs + 1,
            0,
            max_sample - 1,
        ).astype(int)

        k_end = np.clip(
            gate_width // 2 + peak_idxs + 1,
            0,
            max_sample - 1,
        ).astype(int)

        delta = np.bincount(k_start, minlength=max_sample) - np.bincount(
            k_end, minlength=max_sample
        )

        gate_mask = np.cumsum(delta)[:max_sample] > 0
        gated_idxs = np.nonzero(gate_mask)[0]

        return gate_mask, gated_idxs

    def _compute_gate_peak_mapping(
        peak_idxs: np.ndarray,
        gated_idxs: np.ndarray,
        gate_width: int,
        max_sample: int,
    ) -> np.ndarray:
        """Map each gated sample to its corresponding peak index.

        Parameters
        ----------
        peak_idxs : np.ndarray
            The indices of the ECG peaks to gate.
        gated_idxs : np.ndarray
            The indices of the gated samples.
        gate_width : int
            The width of the gates.
        max_sample : int
            The maximum sample index (length of the EMG signal).

        Returns
        -------
        numpy.ndarray
            The indices of the corresponding peaks for each gated sample.


        """
        # Compute the start of each gate window, ensuring they are within
        # bounds
        gate_starts = np.clip(
            -gate_width // 2 + peak_idxs + 1,
            0,
            max_sample,
        ).astype(int)

        # For each gated index, find the index of the last gate start that is
        # less than or equal to it. This gives the corresponding peak index.
        return np.clip(
            np.searchsorted(gate_starts, gated_idxs, side="right") - 1,
            0,
            len(peak_idxs) - 1,
        )

    # Main gating function
    peak_idxs = np.sort(np.asarray(peak_idxs))
    emg_raw_gated = np.array(emg_raw, dtype=float, copy=True)

    # These methods operate on individual peak-centered gate windows.
    if method in (GATE_FILL_ZEROS, GATE_FILL_INTERP, GATE_FILL_PRIOR_MEAN):
        # Determine the indices of the gated samples for methods 0, 1, and 2.
        gated_idxs = _compute_gated_indices(
            peak_idxs, gate_width, emg_raw_gated.shape[0]
        )

        # Method 0: Fill with zeros
        if method == GATE_FILL_ZEROS:
            emg_raw_gated[gated_idxs] = 0
            return emg_raw_gated

        # Methods 1 and 2: Compute mask via scatter grid, remaining linkage
        # between sample and the corresponding peak_nr. As a result, pre- and
        # post-gate values can be determined in a vectorized manner.
        fill_idxs = _compute_gate_peak_mapping(
            peak_idxs, gated_idxs, gate_width, emg_raw_gated.shape[0]
        )

        # Method 1: Fill with interpolation pre- and post gate sample
        if method == GATE_FILL_INTERP:  # 1
            return _gate_fill_interpolation(
                fill_idxs=fill_idxs,
                emg_raw_gated=emg_raw_gated,
                gated_idxs=gated_idxs,
                peak_idxs=peak_idxs,
                gate_width=gate_width,
            )

        # Method 2: Fill with window length mean over prior section
        # ..._____|_______|_______|XXXXXXX|XXXXXXX|_____...
        #         ^               ^- gate start   ^- gate end
        #         - peak - half_gate_width * 3 (replacer)  # noqa: ERA001
        if method == GATE_FILL_PRIOR_MEAN:  # 2
            return _gate_fill_prior_mean(
                emg_raw=emg_raw,
                emg_raw_gated=emg_raw_gated,
                gated_idxs=gated_idxs,
                peak_idxs=peak_idxs,
                gate_width=gate_width,
            )

    # Method 3: Fill with moving average over RMS
    # RMS reconstruction operates on merged gate regions because overlapping
    # gates must be treated as a single continuous interval.
    if method == GATE_FILL_RMS:
        # Build mask for all samples within gate windows around detected peaks.
        gate_mask, gated_idxs = _compute_gate_mask(
            peak_idxs, gate_width, emg_raw_gated.shape[0]
        )

        return _gate_fill_rms(
            emg_raw_gated,
            gate_mask,
            gated_idxs,
            gate_width,
        )

    msg = f"Invalid method {method}."
    raise ValueError(msg)


def wavelet_denoising(
    emg_raw: np.ndarray,
    ecg_peak_idxs: np.ndarray,
    fs: int,
    hard_thresholding: bool = True,
    n: int = 4,
    wavelet_type: str = "db2",
    fixed_threshold: float = 4.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove ECG peaks from EMG signal using wavelet denoising.

    Shrinkage Denoising using a-trous wavelet decomposition (SWT). NB: This
    function assumes that the emg_raw has already been preprocessed for
    removal of baseline, powerline, and aliasing. N.B. This is a Python
    implementation of the SWT, as previously implemented in MATLAB by Jan
    Graßhoff. See Copyright notice below.
    --------------------------------------------------------------------------
    Copyright 2019 Institute for Electrical Engineering in Medicine,
    University of Luebeck
    Jan Graßhoff

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    ---------------------------------------------------------------------------

    Parameters
    ----------
    emg_raw : numpy.ndarray
        1D raw EMG data
    ecg_peak_idxs : numpy.ndarray
        list of R-peaks indices
    fs : int
        Sampling rate of emg_raw
    hard_thresholding : bool, default=True
        True: hard (default), False: soft
    n : int, default=4
        decomposition level
    wavelet_type : str, default='db2'
        wavelet type (see pywt.swt help)
    fixed_threshold : float, default=4.5
        fixed threshold value for wavelet shrinkage

    Returns
    -------
    emg_clean: numpy.ndarray
        emg_clean: cleaned EMG signal

    """

    def estimate_noise(signal: np.ndarray, window_length: int) -> np.ndarray:
        """Estimate noise level.

        Parameters
        ----------
        signal : numpy.ndarray
            wavelet-decomposed signal
        window_length : int
            window length for noise estimation

        Returns
        -------
        std_estimated: numpy.ndarray[float]
            std_estimated: estimated noise level

        """
        nb_level = signal.shape[0]
        std_estimated = np.zeros(signal.shape)

        for k in range(nb_level):
            # Estimate std from MAD: std ~ MAD/0.6745
            std_estimated[k, :] = (
                pd.Series(np.abs(signal[k, :]))
                .rolling(window=window_length, min_periods=1, center=True)
                .median()
                .values
                / 0.6745
            )

            # Correct on- and offset effects
            std_estimated[k, : window_length // 2] = std_estimated[
                k, window_length // 2
            ]
            std_estimated[k, -window_length // 2 :] = std_estimated[
                k, -window_length // 2
            ]
        return std_estimated

    def get_gate_windows(
        rpeak_bool_vec: np.ndarray, window_length: int
    ) -> np.ndarray:
        """Generate gate windows for the peaks.

        Parameters
        ----------
        rpeak_bool_vec : numpy.ndarray
            1D raw, where R-peak location == 1
        window_length : int
            number of samples to gate around peaks

        Returns
        -------
        gate_windows: numpy.ndarray[int]
            gate_windows: gated signal based on R-peaks, where gate == 1

        """
        window_length = int(np.floor(window_length / 2) * 2)
        rpeak_idxs = np.where(rpeak_bool_vec == 1)[0]

        gate_windows = np.zeros_like(rpeak_bool_vec)
        for _, rpeak_idx in enumerate(rpeak_idxs):
            gate_windows[
                max(rpeak_idx - window_length // 2, 0) : min(
                    rpeak_idx + window_length // 2, len(rpeak_bool_vec)
                )
            ] = 1

        return gate_windows

    def threshold_wavelets(
        data: np.ndarray, hard_thresholding: bool, threshold: float
    ) -> np.ndarray:
        """Apply thresholding to data based on 'soft' or 'hard' option.

        Parameters
        ----------
        data : numpy.ndarray
            input data
        hard_thresholding : bool, default=True
            True: hard (default), False: soft
        threshold : float
            threshold value

        Returns
        -------
        data: numpy.ndarray
            data: thresholded data

        """
        if hard_thresholding is True:
            # Hard thresholding
            data[np.abs(data) < threshold] = 0
        elif hard_thresholding is False:
            # Soft thresholding
            data = np.sign(data) * np.maximum(np.abs(data) - threshold, 0)
        return data

    # Calculate gate windows
    r_peak_bool = np.zeros(emg_raw.shape)
    r_peak_bool[ecg_peak_idxs] = 1
    gate_bool_array = get_gate_windows(r_peak_bool, fs // 10)

    # Signal Extension by zero padding
    pow_2_n = 2**n
    n_samp = len(emg_raw)
    n_samp_extended = int(np.ceil(n_samp / pow_2_n) * pow_2_n)
    zero_padding = np.zeros(n_samp_extended - n_samp)
    emg_raw_zero_padded = np.concatenate((emg_raw, zero_padding))
    gate_bool_array = np.concatenate((gate_bool_array, zero_padding))

    # Wavelet decomposition of emg_raw using Stationary Wavelet Transform (SWT)
    wav_dec = pywt.swt(emg_raw_zero_padded, wavelet_type, level=n)
    wav_dec_unpacked = np.array(
        [[subband[0], subband[1]] for subband in wav_dec]
    )
    swc = np.vstack((wav_dec_unpacked[:, 1, :], wav_dec_unpacked[n - 1, 0, :]))

    # Gate out R-peaks in wavelet subbands
    wav_dec_gated = np.array(swc)
    wav_dec_gated[:, gate_bool_array == 1] = np.nan

    # Custom threshold coefficients
    window_length = 15 * fs
    s = estimate_noise(wav_dec_gated[:-1], window_length)

    thresholds = np.zeros_like(swc)
    wxd = np.array(wav_dec_unpacked)

    for k in range(n):
        threshold = float(fixed_threshold * s[k, :])
        thresholds[k, :] = threshold
        wxd[k, 1, :] = threshold_wavelets(
            wav_dec_unpacked[k, 1, :], hard_thresholding, threshold
        )

    # # Wavelet reconstruction
    ecg_reconstructd = np.asarray(
        pywt.iswt([tuple(subband) for subband in wxd], wavelet_type)
    )

    # Return results
    wav_dec = np.array(swc)
    ecg_reconstructd = ecg_reconstructd[:n_samp]
    thresholds = thresholds[:, :n_samp]
    gate_bool_array = gate_bool_array[:n_samp]
    emg_clean = emg_raw - ecg_reconstructd

    return emg_clean, wav_dec, thresholds, gate_bool_array
