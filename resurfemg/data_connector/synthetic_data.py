"""Copyright 2022 Netherlands eScience Center and Twente University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods to create synthetic data with several methods.
"""

import numpy as np
import pandas as pd
from scipy import signal

from resurfemg.preprocessing import filtering as filt

DEFAULT_LUNG_MECHANICS = {
    "c": 0.050,
    "r": 5,
}
DEFAULT_VENT_SETTINGS = {
    "dp": 5,
    "peep": 5,
    "flow_cycle": 0.25,  # Fraction F_max
    "flow_trigger": 2,  # L/min
    "tau_dp_up": 10,
    "tau_dp_down": 5,
}

DEFAULT_VENT_STATUS = {
    "p_set": 0,
    "active": False,
    "sensitive": True,
    "F_max": 0,
}

F_HIGH = 0.05  # Hz


rng = np.random.default_rng()


def respiratory_pattern_generator(
    t_end: float = 7 * 60,
    fs: int = 2048,
    rr: float = 22,
    ie_ratio: float = 1 / 2,
    t_p_occs: float | None = None,
) -> np.ndarray:
    """Simulate on/off respiratory muscle activation pattern.

    This function simulates an on/off respiratory muscle activiation pattern
    for generating a synthetic EMG.
    ---------------------------------------------------------------------------

    Args:
        t_end (float): end time (Default value = 7 * 60)
        fs (int): Sampling rate (Default value = 2048)
        rr (float): Respiratory rate (/min) (Default value = 22)
        ie_ratio (float): Ratio between inspiratory and expiratory time
            (Default value = 1 / 2)
        t_p_occs (float): Timing of occlusions (s) (Default value = None)

    Returns:
        np.array[float]: The simulated on/off respiratory muscle pattern.

    """
    ie_fraction = ie_ratio / (ie_ratio + 1)
    if t_p_occs is None:
        t_occs = np.array([])
    else:
        t_occs = np.floor(np.array(t_p_occs) * rr / 60) * 60 / rr

    for t_occ in t_occs:
        if t_end < (t_occ + 60 / rr):
            print(
                "t="
                + str(t_occ)
                + ":t_occ"
                + "should be at least a full respiratory cycle from t_end."
            )
    # time axis
    t_emg = np.array([i / fs for i in range(int(t_end * fs))])
    pattern_length = fs * 60 / rr  # samples per breath
    n_breath = int(pattern_length) + 1
    # reference signal pattern generator
    respiratory_pattern = (
        signal.square(t_emg * rr / 60 * 2 * np.pi + 0.5, ie_fraction) + 1
    ) / 2
    blocker = np.arange(n_breath) / pattern_length * 2 * np.pi
    squared_wave = (signal.square(blocker, ie_fraction) + 1) / 2
    for t_occ in t_occs:
        i_occ = int(t_occ * fs)
        # if the occlusion is too close to the end of the signal,
        # we will only fill the remaining part of the signal
        seg = respiratory_pattern[i_occ : i_occ + n_breath]
        seg[:] = squared_wave[: len(seg)]
    return respiratory_pattern


def simulate_muscle_dynamics(
    block_pattern: np.ndarray,
    fs: int = 2048,
    tau_mus_up: float = 0.3,
    tau_mus_down: float = 0.3,
) -> np.ndarray:
    """Simulate muscle activation dynamics.

    This function simulates an respiratory muscle activation dynamics for
    generating a synthetic EMG.
    ---------------------------------------------------------------------------

    Args:
      block_pattern(np.array[float]): Simulated on/off respiratory muscle pattern.
      fs(int): Sampling rate (Default value = 2048)
      tau_mus_up(float): Muscle contraction time constant (s) (Default value = 0.3)
      tau_mus_down(float): Muscle relaxation time constant (s) (Default value = 0.3)

    Returns:
      np.array[float]: The simulated muscle activation pattern.

    """
    # simulate up- and downslope dynamics of EMG
    muscle_activation = np.zeros(len(block_pattern))
    for i in range(1, len(block_pattern)):
        pat = muscle_activation[i - 1]
        pat_delta = block_pattern[i - 1] - pat
        if pat_delta > 0:
            muscle_activation[i] = pat + ((pat_delta) / (tau_mus_up * fs))
        else:
            muscle_activation[i] = pat + ((pat_delta) / (tau_mus_down * fs))
    return muscle_activation


def evaluate_ventilator_status(
    idx: int,
    y_vent: np.ndarray,
    vent_settings: dict,
    vent_status: dict,
) -> dict:
    """Derive the ventilator status based on its settings.

    Define the ventilator status (active support, sensitive for trigger)
    based on the ventilator settings and ventilator flow.
    -----------------------------------------------------------------------

    Args:
        idx(int): The index to evaluate the ventilator status
        y_vent(numpy.ndarray): The ventilator pressure, flow and volume
        vent_settings(dict): The ventilator settings
        vent_status(dict): The current ventilator status

    Returns:
        vent_status: dict: vent_status: The new ventilator status

    """
    if (vent_status["sensitive"] is True) and (
        60 * y_vent[1, idx] > vent_settings["flow_trigger"]
    ):
        vent_status["active"] = True
        vent_status["sensitive"] = False

    if vent_status["active"] and y_vent[1, idx] > vent_status["F_max"]:
        vent_status["p_set"] = vent_settings["dp"]
        vent_status["F_max"] = y_vent[1, idx]
    elif (
        y_vent[1, idx] < vent_settings["flow_cycle"] * vent_status["F_max"]
    ) and vent_status["active"]:
        vent_status["active"] = False
        vent_status["p_set"] = 0
    return vent_status


def simulate_ventilator_data(
    p_mus: np.ndarray,
    fs_vent: int = 100,
    t_occ_bool: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Simulate ventilator data with occlusion manouevers.

    This function simulates ventilator data with occlusion manouevers based on
    the provided `p_mus` and adds noise to the signal.
    ---------------------------------------------------------------------------

    Args:
        p_mus(numpy.ndarray[float]): Respiratory muscle pressure
        fs_vent(int): Ventilator sampling rate(Default value = 100)
        t_occ_bool(numpy.ndarray[bool]): Boolean array.
            Is true when a Pocc manoeuver is done (Default value = None)
        **kwargs: additional keyword arguments for lung mechanics and ventilator settings

    Returns:
        np.array[float]: The synthetic ventilator pressure, flow and volume

    """

    def _simulate_airway_pressure() -> np.ndarray:
        # Simulate up- and downslope dynamics of airway pressure

        return (
            0
            * pd.Series(rng.normal(0, 2, size=(len(p_mus),)))
            .rolling(fs_vent, min_periods=1, center=True)
            .mean()
            .values
        )

    lung_mechanics = {
        key: kwargs.pop(key, DEFAULT_LUNG_MECHANICS[key])
        for key in DEFAULT_LUNG_MECHANICS
    }
    vent_settings = {
        key: kwargs.pop(key, DEFAULT_VENT_SETTINGS[key])
        for key in DEFAULT_VENT_SETTINGS
    }
    if kwargs:
        for key in kwargs:
            msg = f"kwarg `{key}` not available."
            raise UserWarning(msg)

    if t_occ_bool is None:
        t_occ_bool = np.zeros_like(p_mus, dtype=bool)

    p_noise_ma = _simulate_airway_pressure()

    p_dp = -p_mus
    y_vent = np.zeros((3, len(p_mus)))
    flow, volume = y_vent[1], y_vent[2]
    vent_status = DEFAULT_VENT_STATUS.copy()
    for i in range(1, len(p_mus)):
        dp_prev = p_dp[i - 1]
        vent_status = evaluate_ventilator_status(
            idx=i - 1,
            y_vent=y_vent,
            vent_settings=vent_settings,
            vent_status=vent_status,
        )
        dp_step = vent_status["p_set"] - (dp_prev + p_noise_ma[i - 1])
        if t_occ_bool[i]:
            vent_status = DEFAULT_VENT_STATUS.copy()

            # Occlusion pressure results into negative airway pressure:
            dp_step = -np.mean(p_mus[i - int(fs_vent / 2) : int(i - 1)]) - dp_prev
            p_dp[i] = dp_prev + dp_step / (vent_settings["tau_dp_up"])
            # During occlusion manoeuvre: flow and volume are zero
            flow[i] = 0
            volume[i] = 0
        else:
            if (vent_status["p_set"] - dp_prev) > 0:
                p_dp[i] = dp_prev + dp_step / vent_settings["tau_dp_up"]
            else:
                p_dp[i] = dp_prev + dp_step / vent_settings["tau_dp_down"]
            volume_prev = volume[i - 1]
            # Calculate flows and volumes from equation of motion:
            flow[i] = flow_current = (
                (dp_prev + p_mus[i - 1]) - volume_prev / lung_mechanics["c"]
            ) / lung_mechanics["r"]
            volume[i] = volume_prev + flow_current / fs_vent
            if (vent_status["sensitive"] is False) and (flow_current < 0):
                vent_status["sensitive"] = True
                vent_status["F_max"] = 0
    y_vent[0] = vent_settings["peep"] + p_dp

    return y_vent


def simulate_emg(
    muscle_activation: np.ndarray,
    fs_emg: int = 2048,
    emg_amp: float = 5,
    drift_amp: float = 100,
    noise_amp: float = 2,
) -> np.ndarray:
    """Simulate surface respiratory EMG.

    This function simulates an surface respiratory emg based on the provided
    `muscle_activation` and adds noise and drift to the signal. No ecg
    component is included, but can be added later.
    ---------------------------------------------------------------------------

    Args:
        muscle_activation(float): The muscle activation pattern
        fs_emg(int): EMG sampling rate (Default value = 2048)
        emg_amp(float): Approximate EMG-RMS amplitude (uV) (Default value = 5)
        drift_amp(float): Approximate drift RMS amplitude (uV) (Default value = 100)
        noise_amp(float): Approximate baseline noise RMS amplitude (uV) (Default value = 2)

    Returns:
        np.array[float]: The raw synthetic EMG without the ECG added.

    """
    n_samp = len(muscle_activation)
    # make respiratory EMG component
    part_emg = muscle_activation * rng.normal(0, 2, size=(n_samp,))

    # make noise and drift components
    part_noise = rng.normal(0, 2 * noise_amp, size=(n_samp,))
    part_drift = np.zeros((n_samp,))

    white_noise = rng.normal(0, drift_amp, size=(n_samp + int(1 / F_HIGH) * fs_emg,))
    part_drift_tmp = filt.emg_lowpass_butter(white_noise, F_HIGH, fs_emg, order=3)
    part_drift = part_drift_tmp[int(1 / F_HIGH) * fs_emg :] / F_HIGH

    # mix channels, could be remixed with an ecg
    return emg_amp * part_emg + part_drift + part_noise
