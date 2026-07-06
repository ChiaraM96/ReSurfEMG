"""This file contains methods to create synthetic data with several methods.

Copyright 2022 Netherlands eScience Center and Twente University.
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import signal

from resurfemg.preprocessing import filtering as filt

logger = logging.getLogger(__name__)
rng = np.random.default_rng(None)


def respiratory_pattern_generator(
    t_end: float = 7 * 60,
    fs: int = 2048,
    rr: float = 22,
    ie_ratio: float = 1 / 2,
    t_p_occs: float | None = None,
) -> np.ndarray:
    """Simulate an on/off respiratory muscle activation pattern.

    This function simulates an on/off respiratory muscle activation pattern
    for generating a synthetic EMG.

    Args:
        t_end (float): End time.
        fs (int): Sampling rate.
        rr (float): Respiratory rate (/min).
        ie_ratio (float): Ratio between inspiratory and expiratory time.
        t_p_occs (float): Timing of occlusions (s).

    Returns:
        np.array[float]: The simulated on/off respiratory muscle pattern.
    """
    ie_fraction = ie_ratio / (ie_ratio + 1)
    t_occs = np.array([]) if t_p_occs is None else np.floor(np.array(t_p_occs) * rr / 60) * 60 / rr

    for _, t_occ in enumerate(t_occs):
        if t_end < (t_occ + 60 / rr):
            msg = f"t={t_occ}:t_occ should be at least a full respiratory cycle from t_end."
            logger.warning(msg)

    # time axis
    t_emg = np.array([i / fs for i in range(int(t_end * fs))])

    # reference signal pattern generator
    respiratory_pattern = (signal.square(t_emg * rr / 60 * 2 * np.pi + 0.5, ie_fraction) + 1) / 2
    for _, t_occ in enumerate(t_occs):
        i_occ = int(t_occ * fs)
        blocker = np.arange(int(fs * 60 / rr) + 1) / fs * rr / 60 * 2 * np.pi
        squared_wave = (signal.square(blocker, ie_fraction) + 1) / 2
        respiratory_pattern[i_occ : i_occ + int(fs * 60 / rr) + 1] = squared_wave
    return np.asarray(respiratory_pattern)


def simulate_muscle_dynamics(
    block_pattern: np.ndarray,
    fs: int = 2048,
    tau_mus_up: float = 0.3,
    tau_mus_down: float = 0.3,
) -> np.ndarray:
    """Simulate respiratory muscle activation dynamics.

    This function simulates respiratory muscle activation dynamics for
    generating a synthetic EMG.

    Args:
        block_pattern (np.array[float]): Simulated on/off respiratory muscle
            pattern.
        fs (int): Sampling rate.
        tau_mus_up (float): Muscle contraction time constant (s).
        tau_mus_down (float): Muscle relaxation time constant (s).

    Returns:
        np.array[float]: The simulated muscle activation pattern.
    """
    # simulate up- and downslope dynamics of EMG
    muscle_activation = np.zeros((len(block_pattern),))
    for i in range(1, len(block_pattern)):
        pat = muscle_activation[i - 1]
        if (block_pattern[i - 1] - pat) > 0:
            muscle_activation[i] = pat + ((block_pattern[i - 1] - pat) / (tau_mus_up * fs))
        else:
            muscle_activation[i] = pat + ((block_pattern[i - 1] - pat) / (tau_mus_down * fs))
    return muscle_activation


def simulate_powerline_noise(
    n_samp: int,
    fs_emg: int = 2048,
    f0: float = 50,
    n_harmonics: int = 11,
    powerline_amp: float = 5,
    harmonic_decay: float = 0.9,
    freq_drift: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """Simulate mains (powerline) interference with harmonics.

    This function simulates 50/60 Hz powerline interference and its harmonics
    for remixing into a synthetic EMG. Harmonics span up to (but not past) the
    Nyquist frequency, with an irregular per-harmonic amplitude profile that
    reproduces the jagged harmonic peaks seen in real recordings.

    Args:
        n_samp (int): Number of samples to generate.
        fs_emg (int): EMG sampling rate.
        f0 (float): Fundamental line frequency (50 in EU, 60 in US).
        n_harmonics (int): Number of components including the fundamental.
            Clamped so no harmonic exceeds the Nyquist frequency.
        powerline_amp (float): Peak amplitude of the fundamental (uV).
        harmonic_decay (float): Baseline relative amplitude of successive
            harmonics; harmonic k gets `harmonic_decay ** (k - 1)`. Close to 1
            keeps high harmonics strong (as in the reference spectrum).
        freq_drift (float): Std (Hz) of a slow, bounded wander of the line
            frequency. 0 keeps it fixed at f0.
        **kwargs: Optional arguments — harmonic_jitter (log-spread of the
            per-harmonic amplitudes, 0 -> smooth geometric decay),
            harmonic_amplitudes (explicit relative amplitudes, overrides decay
            and jitter), amp_modulation, mod_freq, phase_jitter.

    Returns:
        np.array[float]: The synthetic powerline interference.

    Raises:
        UserWarning: If a kwarg is not an available setting.
    """
    settings = {
        "harmonic_jitter": 0.6,
        "harmonic_amplitudes": None,
        "amp_modulation": 0.0,
        "mod_freq": 0.3,
        "phase_jitter": 0.0,
    }
    for key, value in kwargs.items():
        if key in settings:
            settings[key] = value
        else:
            msg = f"kwarg `{key}` not available."
            raise UserWarning(msg)

    # Never place a harmonic above Nyquist (it would alias to a false peak)
    n_harmonics = min(n_harmonics, int((fs_emg / 2) / f0))

    t = np.arange(n_samp) / fs_emg

    # Slowly wandering fundamental frequency -> instantaneous phase
    if freq_drift > 0:
        dev = np.zeros(n_samp)
        for f_wander in (0.05, 0.13, 0.27):  # sub-Hz -> smooth, bounded drift
            dev += np.sin(2 * np.pi * f_wander * t + rng.uniform(0, 2 * np.pi))
        dev *= freq_drift / (dev.std() + 1e-12)
        f_inst = f0 + dev
    else:
        f_inst = np.full(n_samp, f0)
    phase = 2 * np.pi * np.cumsum(f_inst) / fs_emg

    # Harmonic amplitude profile
    if settings["harmonic_amplitudes"] is not None:
        amps = np.asarray(settings["harmonic_amplitudes"], dtype=float)
        n_harmonics = amps.size
    else:
        amps = harmonic_decay ** np.arange(n_harmonics)
        # irregular profile: real mains harmonics are not monotonic
        if settings["harmonic_jitter"] > 0:
            jit = settings["harmonic_jitter"]
            amps *= np.exp(rng.uniform(-jit, jit, n_harmonics))

    # Sum fundamental + harmonics
    powerline = np.zeros(n_samp)
    phase_jitter = settings["phase_jitter"]
    for k in range(1, n_harmonics + 1):
        phi = rng.normal(0, phase_jitter) if phase_jitter > 0 else 0.0
        powerline += amps[k - 1] * np.sin(k * phase + phi)

    if settings["amp_modulation"] > 0:
        powerline *= 1 + settings["amp_modulation"] * np.sin(
            2 * np.pi * settings["mod_freq"] * t + rng.uniform(0, 2 * np.pi)
        )

    return powerline_amp * powerline


def _evaluate_ventilator_status(
    idx: int,
    y_vent: np.ndarray,
    vent_settings: dict,
    vent_status: dict,
) -> dict:
    """Define the ventilator status.

    The status (active support, sensitive for trigger) is based on the
    ventilator settings and ventilator flow.

    Args:
        idx (int): The index to evaluate the ventilator status.
        y_vent (numpy.ndarray): The ventilator pressure, flow and volume.
        vent_settings (dict): The ventilator settings.
        vent_status (dict): The current ventilator status.

    Returns:
        dict: The new ventilator status.
    """
    if (vent_status["sensitive"] is True) and (60 * y_vent[1, idx] > vent_settings["flow_trigger"]):
        vent_status["active"] = True
        vent_status["sensitive"] = False

    if vent_status["active"] and y_vent[1, idx] > vent_status["F_max"]:
        vent_status["p_set"] = vent_settings["dp"]
        vent_status["F_max"] = y_vent[1, idx]
    elif (y_vent[1, idx] < vent_settings["flow_cycle"] * vent_status["F_max"]) and vent_status["active"]:
        vent_status["active"] = False
        vent_status["p_set"] = 0
    return vent_status


def simulate_ventilator_data(
    p_mus: np.ndarray,
    fs_vent: int = 100,
    t_occ_bool: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Simulate ventilator data with occlusion manoeuvres.

    This function simulates ventilator data with occlusion manoeuvres based
    on the provided `p_mus` and adds noise to the signal.

    Args:
        p_mus (numpy.ndarray[float]): Respiratory muscle pressure.
        fs_vent (int): Ventilator sampling rate.
        t_occ_bool (numpy.ndarray[bool]): Boolean array. Is true when a Pocc
            manoeuvre is done.
        **kwargs: Overrides for the lung mechanics and ventilator settings.

    Returns:
        np.array[float]: The synthetic ventilator pressure, flow and volume.

    Raises:
        UserWarning: If a kwarg is not an available lung mechanics or
            ventilator setting.
    """
    lung_mechanics = {
        "c": 0.050,
        "r": 5,
    }
    vent_settings = {
        "dp": 5,
        "peep": 5,
        "flow_cycle": 0.25,  # Fraction F_max
        "flow_trigger": 2,  # L/min
        "tau_dp_up": 10,
        "tau_dp_down": 5,
    }
    for key, value in kwargs.items():
        if key in lung_mechanics:
            lung_mechanics[key] = value
        elif key in vent_settings:
            vent_settings[key] = value
        else:
            msg = f"kwarg `{key}` not available."
            raise UserWarning(msg)

    if t_occ_bool is None:
        t_occ_bool = np.zeros(p_mus.shape, dtype=bool)

    # Simulate up- and downslope dynamics of airway pressure
    p_noise_ma = (
        0
        * pd.Series(rng.normal(0, 2, size=(len(p_mus),))).rolling(fs_vent, min_periods=1, center=True).mean().to_numpy()
    )

    vent_status = {
        "p_set": 0,
        "active": False,
        "sensitive": True,
        "F_max": 0,
    }
    p_dp = -p_mus
    y_vent = np.zeros((3, len(p_mus)))
    for i in range(1, len(p_mus)):
        vent_status = _evaluate_ventilator_status(
            idx=i - 1,
            y_vent=y_vent,
            vent_settings=vent_settings,
            vent_status=vent_status,
        )
        dp_step = vent_status["p_set"] - (p_dp[i - 1] + p_noise_ma[i - 1])
        if t_occ_bool[i]:
            vent_status["active"] = False
            vent_status["sensitive"] = False
            vent_status["p_set"] = 0
            vent_status["F_max"] = 0

            # Occlusion pressure results into negative airway pressure:
            dp_step = -np.mean(p_mus[i - int(1 * fs_vent / 2) : int(i - 1)]) - p_dp[i - 1]
            p_dp[i] = p_dp[i - 1] + dp_step / (vent_settings["tau_dp_up"])
            # During occlusion manoeuvre: flow and volume are zero
            y_vent[1:2, i] = 0
        else:
            if (vent_status["p_set"] - p_dp[i - 1]) > 0:
                p_dp[i] = p_dp[i - 1] + dp_step / vent_settings["tau_dp_up"]
            else:
                p_dp[i] = p_dp[i - 1] + dp_step / vent_settings["tau_dp_down"]
            # Calculate flows and volumes from equation of motion:
            y_vent[1, i] = ((p_dp[i - 1] + p_mus[i - 1]) - y_vent[2, i - 1] / lung_mechanics["c"]) / lung_mechanics["r"]
            y_vent[2, i] = y_vent[2, i - 1] + y_vent[1, i] * 1 / fs_vent
            if (vent_status["sensitive"] is False) and (y_vent[1, i] < 0):
                vent_status["sensitive"] = True
                vent_status["F_max"] = 0
    y_vent[0, :] = vent_settings["peep"] + p_dp

    return y_vent


def simulate_emg(
    muscle_activation: np.ndarray,
    fs_emg: int = 2048,
    emg_amp: float = 5,
    drift_amp: float = 100,
    noise_amp: float = 2,
    powerline_amp: float = 0,
    powerline_kwargs: dict | None = None,
) -> np.ndarray:
    """Simulate a surface respiratory EMG.

    This function simulates a surface respiratory EMG based on the provided
    `muscle_activation` and adds noise, drift, and powerline interference to
    the signal. No ecg component is included, but can be added later.

    Args:
        muscle_activation (np.array[float]): The muscle activation pattern.
        fs_emg (int): EMG sampling rate.
        emg_amp (float): Approximate EMG-RMS amplitude (uV).
        drift_amp (float): Approximate drift RMS amplitude (uV).
        noise_amp (float): Approximate baseline noise RMS amplitude (uV).
        powerline_amp (float): Peak amplitude of the powerline fundamental
            (uV). 0 disables mains interference.
        powerline_kwargs (dict | None): Additional keyword arguments for the powerline noise simulation.

    Returns:
        np.array[float]: The raw synthetic EMG without the ECG added.
    """
    n_samp = len(muscle_activation)
    # make respiratory EMG component
    part_emg = muscle_activation * rng.normal(0, 2, size=(n_samp,))

    # make noise and drift components
    part_noise = rng.normal(0, 2 * noise_amp, size=(n_samp,))
    part_drift = np.zeros((n_samp,))

    f_high = 0.05
    white_noise = rng.normal(0, drift_amp, size=(n_samp + int(1 / f_high) * fs_emg,))
    part_drift_tmp = filt.emg_lowpass_butter(white_noise, f_high, fs_emg, order=3)
    part_drift = part_drift_tmp[int(1 / f_high) * fs_emg :] / f_high

    # make powerline interference component
    part_powerline = np.zeros((n_samp,))
    if powerline_amp > 0:
        part_powerline = simulate_powerline_noise(
            n_samp,
            fs_emg=fs_emg,
            powerline_amp=powerline_amp,
            **(powerline_kwargs or {}),
        )

    # mix channels, could be remixed with an ecg
    return emg_amp * part_emg + part_drift + part_noise + part_powerline
