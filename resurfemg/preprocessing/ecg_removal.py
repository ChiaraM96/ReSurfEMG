"""ECG artifact removal module.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to eliminate ECG artifacts from EMG arrays.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pywt
import scipy

import resurfemg.preprocessing.filtering as filt
from resurfemg.preprocessing import envelope as evl

# Gate fill method constants (avoid magic-number comparisons)
# 0: filled with zeros, 1: interpolation, 2: prior-window mean, 3: RMS running average
_GATE_FILL_ZEROS = 0
_GATE_FILL_INTERP = 1
_GATE_FILL_PRIOR_MEAN = 2
_GATE_FILL_RMS = 3
_GATE_FILL_QUADRATIC = 4


def detect_ecg_peaks(
    ecg_raw: np.ndarray,
    fs: int,
    peak_fraction: float = 0.4,
    peak_width_s: int | None = None,
    peak_distance: int | None = None,
    bp_filter: bool = True,
) -> np.ndarray:
    """Detect ECG peaks in EMG signal.

    Args:
        ecg_raw (numpy.ndarray): ECG signals to detect the ECG peaks in.
        fs (int): Sampling rate of the EMG signals.
        peak_fraction (float): ECG peaks amplitude threshold relative to the
            specified fraction of the min-max values in the ECG signal.
        peak_width_s (int, optional): ECG peaks width threshold in samples.
        peak_distance (int, optional): Minimum time between ECG peaks in samples.
        bp_filter (bool): Bandpass filter the ecg_raw between 1-500 Hz before
            peak detection.

    Returns:
        numpy.ndarray: ECG peak indices.
    """
    if peak_width_s is None:
        peak_width_s = fs // 1000

    if peak_distance is None:
        peak_distance = fs // 3

    if bp_filter:
        lp_cf = min([500, 0.95 * fs / 2])
        ecg_filt = filt.emg_bandpass_butter(ecg_raw, high_pass=1, low_pass=lp_cf, fs_emg=fs)
        ecg_rms = evl.full_rolling_rms(ecg_filt, fs // 200)
    else:
        ecg_rms = evl.full_rolling_rms(ecg_raw, fs // 200)
    max_ecg_rms = np.percentile(ecg_rms, 99)
    min_ecg_rms = np.percentile(ecg_rms, 1)
    peak_height = peak_fraction * (max_ecg_rms - min_ecg_rms)

    ecg_peak_idxs, _ = scipy.signal.find_peaks(ecg_rms, height=peak_height, width=peak_width_s, distance=peak_distance)

    return ecg_peak_idxs


def _windowed_masked_mean(source: np.ndarray, starts: np.ndarray, width: int, max_sample: int) -> np.ndarray:
    """Windowed masked mean.

    Mean over [start, start + width) windows of `source`, ignoring both
    out-of-bounds positions and NaN values via a MaskedArray. Fully-empty
    windows return NaN.

    Args:
        source (numpy.ndarray): 1D array to window over.
        starts (numpy.ndarray): Window start index per row.
        width (int): Window length.
        max_sample (int): Length of source (out-of-bounds bound).

    Returns:
        numpy.ndarray: Per-window mean, NaN where the window is fully masked.
    """
    idx = starts[:, None] + np.arange(width)  # (K, width)
    oob = (idx < 0) | (idx >= max_sample)
    vals = source[np.clip(idx, 0, max_sample - 1)]
    masked = np.ma.array(vals, mask=oob | np.isnan(vals))
    return masked.mean(axis=1).filled(np.nan)  # (K,)


def _get_emg_raw_gated_rms(emg_raw_gated: np.ndarray, gate_mask: np.ndarray, gate_width: int) -> np.ndarray:
    # define this helper for reusability as it's used in both RMS and quadratic methods
    emg_raw_gated_base = copy.deepcopy(emg_raw_gated)
    emg_raw_gated_base[gate_mask] = np.nan
    return evl.full_rolling_rms(emg_raw_gated_base, gate_width)


def _gate_fill_interp(
    k_start: np.ndarray,
    emg_raw_gated: np.ndarray,
    gated_idx: np.ndarray,
    peaks: np.ndarray,
    half_gate_width: int,
    gate_width: int,
    max_sample: int,
) -> np.ndarray:
    if gated_idx.size == 0:
        return emg_raw_gated

    pre_idx = peaks - half_gate_width - 1
    post_idx = peaks + half_gate_width + 1
    pre_in = (pre_idx >= 0) & (pre_idx < max_sample)
    post_in = (post_idx >= 0) & (post_idx < max_sample)
    pre_val = np.where(pre_in, emg_raw_gated[np.clip(pre_idx, 0, max_sample - 1)], 0.0)
    post_val = np.where(post_in, emg_raw_gated[np.clip(post_idx, 0, max_sample - 1)], 0.0)

    owner = np.clip(np.searchsorted(k_start, gated_idx, side="right") - 1, 0, len(peaks) - 1)
    frac = (gated_idx - peaks[owner] + half_gate_width) / float(gate_width)
    emg_raw_gated[gated_idx] = (1.0 - frac) * pre_val[owner] + frac * post_val[owner]
    return emg_raw_gated


def _gate_fill_prior_mean(
    emg_raw: np.ndarray,
    emg_raw_gated: np.ndarray,
    gated_idx: np.ndarray,
    k_start: np.ndarray,
    peaks: np.ndarray,
    half_gate_width: int,
    gate_width: int,
    max_sample: int,
) -> np.ndarray:
    prior_start = peaks - half_gate_width * 3
    clamped = prior_start < 0
    prior_means = _windowed_masked_mean(emg_raw, prior_start, gate_width, max_sample)
    post_means = _windowed_masked_mean(emg_raw, peaks + half_gate_width, gate_width, max_sample)
    fill_means = np.where(clamped, post_means, prior_means)

    owner = np.clip(np.searchsorted(k_start, gated_idx, side="right") - 1, 0, len(peaks) - 1)
    emg_raw_gated[gated_idx] = fill_means[owner]
    return emg_raw_gated


def _gate_fill_rms(
    emg_raw_gated: np.ndarray,
    gate_mask: np.ndarray,
    gated_idx: np.ndarray,
    gate_width: int,
    max_sample: int,
) -> np.ndarray:
    emg_raw_gated_rms = _get_emg_raw_gated_rms(emg_raw_gated, gate_mask, gate_width)

    w = max(1, 2 * int(1.5 * gate_width) + 1)
    closed = "neither" if gate_width % 2 == 0 else "left"
    local_mean = pd.Series(emg_raw_gated_rms).rolling(window=w, min_periods=1, center=True, closed=closed).mean().values

    fill_vals = np.asarray(local_mean[gated_idx])
    interp_mask = np.isnan(fill_vals)
    emg_raw_gated[gated_idx[~interp_mask]] = fill_vals[~interp_mask]

    return _interp_missing_samples(emg_raw_gated, interp_mask, gated_idx, max_sample)


def _interp_missing_samples(
    emg_raw_gated: np.ndarray,
    interp_mask: np.ndarray,
    sample_idxs: np.ndarray,
    max_sample: int,
) -> np.ndarray:
    interpolate_samples = sample_idxs[interp_mask]
    if interpolate_samples.size == 0:
        return emg_raw_gated

    # Boundary handling: pin a gated endpoint to 0 and PROMOTE it to an anchor
    # (drop it from the interpolation targets) so interior samples ramp from 0,
    # instead of being overwritten and flat-clamped to the first interior value.
    if interpolate_samples[0] == 0:
        emg_raw_gated[0] = 0.0
        interpolate_samples = interpolate_samples[1:]
    if interpolate_samples.size and interpolate_samples[-1] == max_sample - 1:
        emg_raw_gated[-1] = 0.0
        interpolate_samples = interpolate_samples[:-1]
    if interpolate_samples.size == 0:
        return emg_raw_gated

    x_samp = np.arange(max_sample)
    other_mask = ~np.isin(x_samp, interpolate_samples)
    if not other_mask.any():  # no anchors left to interpolate from
        return emg_raw_gated

    emg_raw_gated[interpolate_samples] = np.interp(interpolate_samples, x_samp[other_mask], emg_raw_gated[other_mask])
    return emg_raw_gated


def _gate_fill_quadratic(
    emg_raw_gated: np.ndarray,
    gate_mask: np.ndarray,
    gated_idx: np.ndarray,
    gate_width: int,
    max_sample: int,
    poly_width: int = 5 * 205,
) -> np.ndarray:
    emg_raw_gated_rms = _get_emg_raw_gated_rms(emg_raw_gated, gate_mask, gate_width)
    emg_raw_gated_rms[gate_mask] = np.nan
    # Seed the gate region to NaN so that segments the fit can't cover remain
    # NaN and get caught by the interp fallback below. Support is read from the
    # RMS array, not from emg_raw_gated, so blanking the signal here is safe.
    emg_raw_gated[gate_mask] = np.nan
    # vectorizing this block actually slows the code down, so we keep it as a loop over segments
    if gated_idx.size > 0:
        breaks = np.diff(gated_idx) != 1
        seg_starts = gated_idx[np.concatenate(([True], breaks))]
        seg_ends = gated_idx[np.concatenate((breaks, [True]))]
        for gs, ge in zip(seg_starts, seg_ends, strict=False):
            pre_s = max(0, gs - poly_width // 2)
            post_e = min(max_sample, ge + 1 + poly_width // 2)

            # support indices and values from RMS (skip NaNs)
            idx_pre = np.arange(pre_s, gs)
            idx_post = np.arange(ge + 1, post_e)
            support_idx = np.concatenate((idx_pre, idx_post))
            support_vals = emg_raw_gated_rms[support_idx]
            valid_mask = ~np.isnan(support_vals)
            valid_idx = support_idx[valid_mask]
            valid_vals = support_vals[valid_mask]

            # fit polynomial to valid RMS support
            fit_available = valid_idx.size >= 2  # noqa: PLR2004
            if fit_available:
                deg = min(2, valid_idx.size - 1)
                p_seg = np.polyfit(valid_idx - gs, valid_vals, deg)  # center x on gs
                ks = np.arange(gs, ge + 1)
                emg_raw_gated[gs : ge + 1] = np.polyval(p_seg, ks - gs)  # evaluate on same offset
    interp_mask = np.isnan(emg_raw_gated[gated_idx])
    # now interpolate any remaining NaN samples
    return _interp_missing_samples(emg_raw_gated, interp_mask, gated_idx, max_sample)


def gating_vectorized(
    emg_raw: np.ndarray, peak_idxs: list | np.ndarray, gate_width: int = 205, method: int = 3, **kwargs
) -> np.ndarray:
    """Gating removal of QRS complexes.

    Eliminate peaks (e.g. QRS) from emg_raw using gates
    of width gate_width. The gate either filled by zeros or interpolation.
    The filling method for the gate is encoded as follows:
    0: Filled with zeros
    1: Interpolation samples before and after
    2: Fill with average of prior segment if exists
    otherwise fill with post segment
    3: Fill with running average of RMS (default)

    Args:
        emg_raw (numpy.ndarray): Signal to process.
        peak_idxs (list or numpy.ndarray): List of individual peak index places to
            be gated.
        gate_width (int): Width of the gate.
        method (int): Filling method of gate.
        **kwargs: Additional keyword arguments for interpolation.

    Returns:
        numpy.ndarray: The gated result.
    """
    # all methods require the gate mask, so we can compute it here to avoid rewriting the same logic in each method
    peaks = np.sort(np.asarray(peak_idxs))
    emg_raw_gated = copy.deepcopy(emg_raw)
    max_sample = emg_raw_gated.shape[0]
    # the half_gate_width is computed via integer division for methods 0, 1, and 2,
    # but via float division for methods 3 and 4. The rest of the mask computation is identical for all methods.
    half_gate_width = gate_width / 2 if method in (_GATE_FILL_RMS, _GATE_FILL_QUADRATIC) else gate_width // 2
    k_start = np.clip(peaks - half_gate_width, 0, max_sample).astype(int)
    k_end = np.clip(peaks + half_gate_width, 0, max_sample).astype(int)
    delta = np.zeros(max_sample + 1, dtype=np.int64)
    np.add.at(delta, k_start, 1)  # +1 entering each gate
    np.add.at(delta, k_end, -1)  # -1 leaving each gate
    gate_mask = np.cumsum(delta)[:max_sample] > 0

    # Each method gets its own helper function to improve code readability
    # and reduce the complexity of the main gating function.
    if method == _GATE_FILL_ZEROS:  # 0
        emg_raw_gated[gate_mask] = 0
        return emg_raw_gated

    # again, this is a common step for all non-zero methods, so we can compute it once here.
    gated_idx = np.nonzero(gate_mask)[0]

    if method == _GATE_FILL_INTERP:  # 1
        emg_raw_gated = _gate_fill_interp(
            k_start,
            emg_raw_gated,
            gated_idx,
            peaks,
            int(half_gate_width),
            gate_width,
            max_sample,
        )
    elif method == _GATE_FILL_PRIOR_MEAN:  # 2
        emg_raw_gated = _gate_fill_prior_mean(
            emg_raw,
            emg_raw_gated,
            gated_idx,
            k_start,
            peaks,
            int(half_gate_width),
            gate_width,
            max_sample,
        )
    elif method == _GATE_FILL_RMS:  # 3
        emg_raw_gated = _gate_fill_rms(emg_raw_gated, gate_mask, gated_idx, gate_width, max_sample)
    elif method == _GATE_FILL_QUADRATIC:  # 4
        poly_width = kwargs.get("poly_width", 5 * gate_width)
        emg_raw_gated = _gate_fill_quadratic(emg_raw_gated, gate_mask, gated_idx, gate_width, max_sample, poly_width)
    else:
        msg = f"Invalid method {method}."
        raise ValueError(msg)

    return emg_raw_gated


def gating_original(emg_raw, peak_idxs, gate_width=205, method=1, **kwargs):  # noqa: ANN001, ANN201, C901, PLR0912, PLR0915
    """Eliminate peaks (e.g. QRS) from emg_raw using gates
    of width gate_width. The gate either filled by zeros or interpolation.
    The filling method for the gate is encoded as follows:
    0: Filled with zeros
    1: Interpolation samples before and after
    2: Fill with average of prior segment if exists
    otherwise fill with post segment
    3: Fill with running average of RMS (default).
    ---------------------------------------------------------------------------
    :param emg_raw: Signal to process
    :type emg_raw: ~numpy.ndarray
    :param peak_idxs: list of individual peak index places to be gated
    :type peak_idxs: ~list
    :param gate_width: width of the gate
    :type gate_width: int
    :param method: filling method of gate
    :type method: int

    :returns emg_raw_gated: the gated result
    :rtype emg_raw_gated: numpy.ndarray
    """  # noqa: D205
    emg_raw_gated = copy.deepcopy(emg_raw)
    max_sample = emg_raw_gated.shape[0]
    half_gate_width = gate_width // 2
    if method == 0:
        # Method 0: Fill with zeros
        # Vectorized construction of gate sample indices
        peaks = np.asarray(peak_idxs, dtype=int).ravel()
        if peaks.size == 0:
            gate_samples = np.array([], dtype=int)
        else:
            starts = np.maximum(0, peaks - half_gate_width)
            ends = np.minimum(max_sample, peaks + half_gate_width)
            # build ranges per peak and concat, then unique to avoid duplicates
            gate_samples = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])  # noqa: B905
            gate_samples = np.unique(gate_samples).astype(int)

        emg_raw_gated[gate_samples] = 0
    elif method == 1:
        # Method 1: Fill with interpolation pre- and post gate sample
        peaks = np.asarray(peak_idxs, dtype=int).ravel()
        if peaks.size > 0:
            max_sample = emg_raw_gated.shape[0]
            for peak in peaks:
                pre_idx = peak - half_gate_width - 1
                pre_val = emg_raw[pre_idx] if (pre_idx >= 0 and pre_idx < max_sample) else 0

                post_idx = peak + half_gate_width + 1
                post_val = emg_raw[post_idx] if (post_idx >= 0 and post_idx < max_sample) else 0

                k_start = max(0, peak - half_gate_width)
                k_end = min(peak + half_gate_width, max_sample)
                if k_start >= k_end:
                    continue

                ks = np.arange(k_start, k_end)
                frac = (ks - peak + half_gate_width) / float(gate_width)
                emg_raw_gated[ks] = (1.0 - frac) * pre_val + frac * post_val

    elif method == 2:  # noqa: PLR2004
        # Method 2: Fill with window length mean over prior section
        # ..._____|_______|_______|XXXXXXX|XXXXXXX|_____...
        #         ^               ^- gate start   ^- gate end
        #         - peak - half_gate_width * 3 (replacer)  # noqa: ERA001

        for _, peak in enumerate(peak_idxs):
            start = peak - half_gate_width * 3
            if start < 0:
                start = peak + half_gate_width
            end = start + gate_width
            pre_ave_emg = np.nanmean(emg_raw[start:end])

            k_start = max(0, peak - half_gate_width)
            k_end = min(peak + half_gate_width, emg_raw_gated.shape[0])
            for k in range(k_start, k_end):
                emg_raw_gated[k] = pre_ave_emg

    elif method == 3:  # noqa: PLR2004
        # Method 3: Fill with moving average over RMS
        peaks = np.asarray(peak_idxs, dtype=int).ravel()
        if peaks.size > 0:
            # Build boolean mask of gate samples
            gate_mask = np.zeros(max_sample, dtype=bool)
            half_w = gate_width / 2
            starts = np.maximum(0, (peaks - half_w).astype(int))
            ends = np.minimum(max_sample, (peaks + half_w).astype(int))
            for s, e in zip(starts, ends):  # noqa: B905
                if s < e:
                    gate_mask[s:e] = True

            # Compute RMS with gated samples as NaN
            emg_raw_gated_base = emg_raw_gated.copy()
            emg_raw_gated_base[gate_mask] = np.nan
            emg_raw_gated_rms = evl.full_rolling_rms(emg_raw_gated_base, gate_width)

            # Compute local mean of the RMS over a window of ~3*gate_widths
            w = max(1, 2 * int(1.5 * gate_width) + 1)
            if gate_width % 2 == 0:  # noqa: SIM108
                closed = "neither"
            else:
                closed = "left"
            local_mean = (
                pd.Series(emg_raw_gated_rms).rolling(window=w, min_periods=1, center=True, closed=closed).mean().values
            )

            # Assign local_mean where available within gates
            mask_available = gate_mask & (~np.isnan(local_mean))
            emg_raw_gated[mask_available] = local_mean[mask_available]

            # Remaining gate samples to interpolate
            interp_mask = gate_mask & np.isnan(local_mean)

    elif method == 4:  # noqa: PLR2004
        # Method 4: Fill with quadratic fit to RMS
        peaks = np.asarray(peak_idxs, dtype=int).ravel()
        if peaks.size > 0:
            # Build boolean mask of gate samples
            gate_mask = np.zeros(max_sample, dtype=bool)
            half_w = gate_width / 2
            starts = np.maximum(0, (peaks - half_w).astype(int))
            ends = np.minimum(max_sample, (peaks + half_w).astype(int))
            for s, e in zip(starts, ends):  # noqa: B905
                if s < e:
                    gate_mask[s:e] = True

            # Compute RMS with gated samples as NaN
            emg_raw_gated_base = emg_raw_gated.copy()
            emg_raw_gated_base[gate_mask] = np.nan
            emg_raw_gated_rms = evl.full_rolling_rms(emg_raw_gated_base, gate_width)
            emg_raw_gated_rms[gate_mask] = np.nan

            # Fit a local cubic (or lower-degree if needed) to RMS using
            local_mean = np.full(max_sample, np.nan)
            # identify contiguous gated segments
            gated_idx = np.nonzero(gate_mask)[0]
            if gated_idx.size > 0:
                # find breaks in the gated indices to get segments
                breaks = np.where(np.diff(gated_idx) != 1)[0]
                seg_starts = gated_idx[np.concatenate(([0], breaks + 1))]
                seg_ends = gated_idx[np.concatenate((breaks, [gated_idx.size - 1]))]

                poly_width = kwargs.get("poly_width", 5 * gate_width)

                for gs, ge in zip(seg_starts, seg_ends):  # noqa: B905
                    pre_s = max(0, gs - poly_width // 2)
                    post_e = min(max_sample, ge + 1 + poly_width // 2)

                    # support indices and values from RMS (skip NaNs)
                    idx_pre = np.arange(pre_s, gs)
                    idx_post = np.arange(ge + 1, post_e)
                    support_idx = np.concatenate((idx_pre, idx_post))
                    support_vals = emg_raw_gated_rms[support_idx]
                    valid_mask = ~np.isnan(support_vals)
                    valid_idx = support_idx[valid_mask]
                    valid_vals = support_vals[valid_mask]

                    # fit polynomial to valid RMS support
                    fit_available = valid_idx.size >= 2  # noqa: PLR2004
                    if fit_available:
                        deg = min(2, valid_idx.size - 1)
                        p_seg = np.polyfit(valid_idx, valid_vals, deg)
                        ks = np.arange(gs, ge + 1)
                        emg_raw_gated[gs : ge + 1] = np.polyval(p_seg, ks)
            interp_mask = gate_mask & np.isnan(emg_raw_gated)
    if method in [3, 4]:  # noqa: SIM102
        if np.any(interp_mask):  # type: ignore  # noqa: PGH003
            interp_idx = np.nonzero(interp_mask)[0]  # type: ignore  # noqa: PGH003
            other_idx = np.nonzero(~interp_mask)[0]  # type: ignore  # noqa: PGH003

            # Boundary handling: Set endpoints to 0 if part of interp_idx
            if 0 in interp_idx:
                emg_raw_gated[0] = 0.0
                interp_mask[0] = False  # type: ignore  # noqa: PGH003
            if (max_sample - 1) in interp_idx:
                emg_raw_gated[-1] = 0.0
                interp_mask[-1] = False  # type: ignore  # noqa: PGH003

            interp_idx = np.nonzero(interp_mask)[0]  # type: ignore  # noqa: PGH003
            other_idx = np.nonzero(~interp_mask)[0]  # type: ignore  # noqa: PGH003

            if other_idx.size > 0 and interp_idx.size > 0:
                # Ensure there are values to interpolate from
                emg_raw_gated_interp = np.interp(interp_idx, other_idx, emg_raw_gated[other_idx])
                emg_raw_gated[interp_idx] = emg_raw_gated_interp

    return emg_raw_gated


def gating(
    emg_raw: np.ndarray,
    peak_idxs: list | np.ndarray,
    gate_width: int = 205,
    method: int = 3,
    vectorized: bool = False,
    **kwargs,
) -> np.ndarray:
    """Gating removal of QRS complexes.

    Eliminate peaks (e.g. QRS) from emg_raw using gates
    of width gate_width. The gate either filled by zeros or interpolation.
    The filling method for the gate is encoded as follows:
    0: Filled with zeros
    1: Interpolation samples before and after
    2: Fill with average of prior segment if exists
    otherwise fill with post segment
    3: Fill with running average of RMS (default)

    Args:
        emg_raw (numpy.ndarray): Signal to process.
        peak_idxs (list or numpy.ndarray): List of individual peak index places to
            be gated.
        gate_width (int): Width of the gate.
        method (int): Filling method of gate.
        vectorized (bool): Whether to use the vectorized implementation (default: False).
        **kwargs: Additional keyword arguments for interpolation.

    Returns:
        numpy.ndarray: The gated result.
    """
    return (
        gating_vectorized(emg_raw, peak_idxs, gate_width, method, **kwargs)
        if vectorized
        else gating_original(emg_raw, peak_idxs, gate_width, method, **kwargs)
    )


def wavelet_denoising(
    emg_raw: np.ndarray,
    ecg_peak_idxs: np.ndarray,
    fs: int,
    hard_thresholding: bool = True,
    n: int = 4,
    wavelet_type: str = "db2",
    fixed_threshold: float = 4.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wavelet denoising of ECG artifacts.

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

    Args:
        emg_raw (numpy.ndarray): 1D raw EMG data.
        ecg_peak_idxs (numpy.ndarray): List of R-peak indices.
        fs (int): Sampling rate of emg_raw.
        hard_thresholding (bool): True for hard thresholding (default), False for soft.
        n (int): Decomposition level (default: 4).
        wavelet_type (str): Wavelet type (default: "db2", see pywt.swt help).
        fixed_threshold (float): Fixed threshold multiplier for wavelet coefficients.

    Returns:
        tuple:
            - numpy.ndarray: Cleaned EMG signal.
            - numpy.ndarray: Wavelet decomposition.
            - numpy.ndarray: Threshold values.
            - numpy.ndarray: Gated signal based on R-peaks, where gate == 1.
    """

    def estimate_noise(signal: np.ndarray, window_length: int) -> np.ndarray:
        """Estimate noise level.

        Args:
                signal (numpy.ndarray): Wavelet-decomposed signal.
                window_length (int): Window length for noise estimation.

        Returns:
                numpy.ndarray: Estimated noise level.
        """
        nb_level = signal.shape[0]
        std_estimated = np.zeros(signal.shape)

        for k in range(nb_level):
            # Estimate std from MAD: std ~ MAD/0.6745
            std_estimated[k, :] = (
                pd.Series(np.abs(signal[k, :]))
                .rolling(window=window_length, min_periods=1, center=True)
                .median()
                .to_numpy(dtype=float)
                / 0.6745
            )

            # Correct on- and offset effects
            std_estimated[k, : window_length // 2] = std_estimated[k, window_length // 2]
            std_estimated[k, -window_length // 2 :] = std_estimated[k, -window_length // 2]
        return std_estimated

    def get_gate_windows(rpeak_bool_vec: np.ndarray, window_length: int) -> np.ndarray:
        """Generate gate windows for the peaks.

        Args:
                rpeak_bool_vec (numpy.ndarray): 1D array, where R-peak location == 1.
                window_length (int): Number of samples to gate around peaks.

        Returns:
                numpy.ndarray: Gated signal based on R-peaks, where gate == 1.
        """
        window_length = int(np.floor(window_length / 2) * 2)
        rpeak_idxs = np.where(rpeak_bool_vec == 1)[0]

        gate_windows = np.zeros_like(rpeak_bool_vec)
        for _, rpeak_idx in enumerate(rpeak_idxs):
            gate_windows[
                max(rpeak_idx - window_length // 2, 0) : min(rpeak_idx + window_length // 2, len(rpeak_bool_vec))
            ] = 1

        return gate_windows

    def threshold_wavelets(data: np.ndarray, hard_thresholding: bool, threshold: float | np.ndarray) -> np.ndarray:
        """Threshold wavelet coefficients.

        Apply thresholding to data based on 'soft' or 'hard' option.

        Args:
                data (numpy.ndarray): Input data.
                hard_thresholding (bool): True for hard thresholding, False for soft.
                threshold (float): Threshold value.

        Returns:
                numpy.ndarray: Thresholded data.
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
    wav_dec_unpacked = np.array([[subband[0], subband[1]] for subband in wav_dec])
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
        threshold = fixed_threshold * s[k, :]
        thresholds[k, :] = threshold
        wxd[k, 1, :] = threshold_wavelets(wav_dec_unpacked[k, 1, :], hard_thresholding, threshold)

    # # Wavelet reconstruction
    ecg_reconstructd = pywt.iswt([tuple(subband) for subband in wxd], wavelet_type)

    # Return results
    wav_dec = np.array(swc)
    ecg_reconstructd = ecg_reconstructd[:n_samp]
    thresholds = thresholds[:, :n_samp]
    gate_bool_array = gate_bool_array[:n_samp]
    emg_clean = emg_raw - ecg_reconstructd

    return emg_clean, wav_dec, thresholds, gate_bool_array
