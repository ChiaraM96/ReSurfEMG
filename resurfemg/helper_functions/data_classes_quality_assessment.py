"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains the quality assessment methods for data classes.
"""

from typing import Literal

import numpy as np
import pandas as pd

from resurfemg.data_connector.data_classes import TimeSeries
from resurfemg.data_connector.peakset_class import PeaksSet
from resurfemg.postprocessing import quality_assessment as qa


def initialize_emg_tests(
    timeseries: TimeSeries,
    peak_set_name: str,
    cutoff: Literal["tolerant", "strict"] | dict[str, float] | None = None,
    skip_tests: list[str] | None = None,
    parameter_names: dict[str, str] | None = None,
) -> tuple[list[str], dict, PeaksSet, dict[str, str], int, pd.DataFrame, pd.DataFrame]:
    """Initialize local parameters.

    See TimeSeries.test_emg_quality method in
    resurfemg.data_connector.data_classes for more information.

    Args:
        timeseries (TimeSeries): The time series data.
        peak_set_name (str): The name of the peaks set to test.
        cutoff (Literal["tolerant", "strict"] | dict | None): the cut-off values
            for the test, passed either as a dictionary or as a string indicating
            the type of cut-off to use.
            If passing the dictionary, the keys should be the names of the tests
            and the values should be the cut-off values.
            If passing a string, it should be either "tolerant" or "strict", which
            will use pre-defined cut-off values for each test. If None, default
            "tolerant" cut-off values will be used.
            (Default value = None)
        skip_tests (list[str] | None): the names of the tests to skip.
            (Default value = None)
        parameter_names (dict[str, str] | None): the names by which the parameters
            are known in the provided TimeSeries object.
            Example: ```{"baseline": "slopesum_baseline", "ecg": "rr_peaks",
            "time_product": "electrical_time_product"}```.
            (Default value = None)

    Returns:
        tuple[list[str], dict, PeaksSet, dict[str, str], int, pd.DataFrame, pd.DataFrame]:
        A tuple containing the following elements:
            list[str]: the names of the tests that were skipped.
            dict: the cut-off values for the tests.
            PeaksSet: the PeaksSet object corresponding to the provided peak_set_name.
            dict[str, str]: the names of the parameters in the TimeSeries object.
            int: the number of peaks in the PeaksSet.
            pd.DataFrame: the DataFrame containing the quality values.
            pd.DataFrame: the DataFrame containing the quality outcomes.
    """
    if peak_set_name in timeseries.peaks:
        peak_set = timeseries.peaks[peak_set_name]
    else:
        msg = "Non-existent PeaksSet key"
        raise KeyError(msg)

    if skip_tests is None:
        skip_tests = []

    if cutoff is None or (isinstance(cutoff, str) and cutoff == "tolerant"):
        cutoff = {
            "interpeak_distance": 1.1,
            "snr": 1.4,
            "aub": 40,
            "curve_fit": 30,
            "aub_window_s": 5 * timeseries.param["fs"],
            "bell_window_s": 5 * timeseries.param["fs"],
            "relative_aub_percentile": 75.0,
            "relative_aub_factor": 4.0,
            "relative_etp_upper_percentile": 95.0,
            "relative_etp_upper_factor": 10.0,
            "relative_etp_lower_percentile": 5.0,
            "relative_etp_lower_factor": 0.1,
        }
    elif isinstance(cutoff, str) and cutoff == "strict":
        cutoff = {
            "interpeak_distance": 1.1,
            "snr": 1.75,
            "aub": 30,
            "curve_fit": 25,
            "aub_window_s": 5 * timeseries.param["fs"],
            "bell_window_s": 5 * timeseries.param["fs"],
            "relative_aub_percentile": 75.0,
            "relative_aub_factor": 2.0,
            "relative_etp_upper_percentile": 95.0,
            "relative_etp_upper_factor": 10.0,
            "relative_etp_lower_percentile": 5.0,
            "relative_etp_lower_factor": 0.1,
        }
    elif isinstance(cutoff, dict):
        tests = ["interpeak_distance", "snr", "aub", "curve_fit"]
        for _, test in enumerate(tests):
            if test not in skip_tests:
                if test not in cutoff:
                    raise KeyError("No cut-off value provided for: " + test)
                if isinstance(cutoff, float):
                    raise ValueError("Invalid cut-off value provided for: " + test)

        if "aub" not in skip_tests and "aub_window_s" not in cutoff:
            raise KeyError("No area under the baseline window provided for: " + test)
        if "relative_aub" not in skip_tests:
            if "relative_aub_percentile" not in cutoff:
                msg_0 = (
                    "No relative area under the baseline percentile"
                    " provided for: relative_aub"
                )
                raise KeyError(msg_0)
            if "relative_aub_factor" not in cutoff:
                msg_0 = (
                    "No relative area under the baseline factor "
                    "provided for: relative_aub"
                )
                raise KeyError(msg_0)
        if "relative_etp" not in skip_tests:
            _params = [
                "upper_percentile",
                "upper_factor",
                "lower_percentile",
                "lower_factor",
            ]
            for _param in _params:
                if ("relative_etp_" + _param) not in cutoff:
                    msg_0 = f"No relative_etp_{_param} provided for: relative_aub"
                    raise KeyError(msg_0)

    if parameter_names is None:
        parameter_names = {}

    for parameter in ["ecg", "time_product", "baseline", "ecg"]:
        if parameter not in parameter_names:
            parameter_names[parameter] = parameter

    n_peaks = len(peak_set.peak_df["peak_idx"].to_numpy())
    quality_values_df = pd.DataFrame(
        data=peak_set.peak_df["peak_idx"], columns=pd.Index(["peak_idx"])
    )
    quality_outcomes_df = pd.DataFrame(
        data=peak_set.peak_df["peak_idx"], columns=pd.Index(["peak_idx"])
    )
    return (
        skip_tests,
        cutoff,
        peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df,
    )


def test_interpeak_distance(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    n_peaks: int,
    cutoff: dict[str, float],
    parameter_names: dict[str, str],
) -> pd.DataFrame:
    """Test interpeak distance.

    See TimeSeries.test_emg_quality method in
    resurfemg.data_connector.data_classes for more information.

    Args:
        timeseries (TimeSeries): The time series data.
        peak_set (PeaksSet): The peaks set to test.
        quality_outcomes_df (pd.DataFrame): The DataFrame containing the
            outcomes of the quality tests
        n_peaks (int): the number of peaks in the PeaksSet.
        cutoff (dict[str, float]): the cut-off values for the test, passed as
            a dictionary specifying the cut-off values for each test.
        parameter_names (dict[str, str] | None): the names by which the parameters
            are known in the provided TimeSeries object.
            Example: ```{"baseline": "slopesum_baseline", "ecg": "rr_peaks",
            "time_product": "electrical_time_product"}```.
            (Default value = None)

    Returns:
        pd.DataFrame: the DataFrame containing the quality outcomes after
            testing interpeak distance.

    """
    if parameter_names["ecg"] not in timeseries.peaks:
        msg = (
            "ECG peaks not determined, but required for interpeak distance evaluation."
        )
        raise ValueError(msg)
    ecg_peaks = timeseries.peaks[parameter_names["ecg"]].peak_df["peak_idx"].to_numpy()
    valid_interpeak = qa.interpeak_dist(
        ecg_peak_idxs=ecg_peaks,
        emg_peak_idxs=peak_set.peak_df["peak_idx"].to_numpy(),
        threshold=cutoff["interpeak_distance"],
    )

    valid_interpeak_vec = np.array(n_peaks * [valid_interpeak])
    quality_outcomes_df["interpeak_distance"] = valid_interpeak_vec
    return quality_outcomes_df


def test_snr(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    cutoff: dict[str, float],
    parameter_names: dict[str, str],
) -> pd.DataFrame:
    """Test signal-to-noise ratio.

    See TimeSeries.test_emg_quality method in
    resurfemg.data_connector.data_classes for more information.

    Args:
        timeseries (TimeSeries): The time series data.
        peak_set (PeaksSet): The peaks set to test.
        quality_outcomes_df (pd.DataFrame): The DataFrame containing the
            outcomes of the quality tests
        quality_values_df (pd.DataFrame): The DataFrame containing the values
            of the quality metrics that were tested.
        cutoff (dict[str, float]): the cut-off values for the test, passed as
            a dictionary specifying the cut-off values for each test.
        parameter_names (dict[str, str] | None): the names by which the parameters
            are known in the provided TimeSeries object.
            Example: ```{"baseline": "slopesum_baseline", "ecg": "rr_peaks",
            "time_product": "electrical_time_product"}```.
            (Default value = None)

    Returns:
        pd.DataFrame: the DataFrame containing the quality outcomes after
            testing signal-to-noise ratio.

    """
    if parameter_names["baseline"] not in timeseries._y_data:
        msg = "Baseline not determined, but required for  SNR evaluaton."
        raise ValueError(msg)
    snr_peaks = qa.snr_pseudo(
        src_signal=peak_set.signal,
        peaks=peak_set.peak_df["peak_idx"].to_numpy(),
        baseline=timeseries[parameter_names["baseline"]],
        fs=timeseries.param["fs"],
    )
    quality_values_df["snr"] = snr_peaks
    valid_snr = snr_peaks > cutoff["snr"]
    quality_outcomes_df["snr"] = valid_snr
    return quality_outcomes_df


def test_aub(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    cutoff: dict[str, float],
    parameter_names: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test percentage area under the baseline.

    See TimeSeries.test_emg_quality
    method in resurfemg.data_connector.data_classes for more information.

    Args:
      timeseries (TimeSeries):
      peak_set (PeaksSet):
      quality_outcomes_df: pd.DataFrame:
      quality_values_df: pd.DataFrame:
      cutoff (dict[str, float]):
      parameter_names (dict[str, str]):

    Returns:

    """
    if parameter_names["baseline"] not in timeseries._y_data:
        msg = (
            "Baseline not determined, but required for "
            " area under the baseline (AUB) evaluaton."
        )
        raise ValueError(msg)
    if "start_idx" not in peak_set.peak_df.columns:
        msg = (
            "start_idxs not determined, but required for "
            " area under the baseline (AUB) evaluaton."
        )
        raise ValueError(msg)
    if "end_idx" not in peak_set.peak_df.columns:
        msg = (
            "end_idxs not determined, but required for "
            " area under the baseline (AUB) evaluaton."
        )
        raise ValueError(msg)
    outputs = qa.percentage_under_baseline(
        signal=peak_set.signal,
        fs=timeseries.param["fs"],
        peak_idxs=peak_set.peak_df["peak_idx"].to_numpy(),
        start_idxs=peak_set.peak_df["start_idx"].to_numpy(),
        end_idxs=peak_set.peak_df["end_idx"].to_numpy(),
        baseline=timeseries[parameter_names["baseline"]],
        aub_window_s=None,
        ref_signal=None,
        aub_threshold=cutoff["aub"],
    )

    (valid_timeproducts, percentages_aub, y_refs) = outputs
    quality_values_df["aub"] = percentages_aub
    quality_values_df["aub_y_refs"] = y_refs
    quality_outcomes_df["aub"] = valid_timeproducts
    return quality_outcomes_df, quality_values_df


def test_curve_fits(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    cutoff: dict[str, float],
    parameter_names: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, PeaksSet]:
    """Test curve fit.

    See TimeSeries.test_emg_quality method in
    resurfemg.data_connector.data_classes for more information.

    Args:
      timeseries (TimeSeries):
      peak_set (PeaksSet):
      quality_outcomes_df: pd.DataFrame:
      quality_values_df: pd.DataFrame:
      cutoff (dict[str, float]):
      parameter_names (dict[str, str]):

    Returns:

    """
    if timeseries.baseline is None:
        msg = (
            "Baseline not determined, but required for "
            " area under the baseline (AUB) evaluaton."
        )
        raise ValueError(msg)
    if "start_idx" not in peak_set.peak_df.columns:
        msg = "start_idxs not determined, but required for  curve fit evaluaton."
        raise ValueError(msg)
    if "end_idx" not in peak_set.peak_df.columns:
        msg = "end_idxs not determined, but required for  curve fit evaluaton."
        raise ValueError(msg)
    if parameter_names["time_product"] not in peak_set.peak_df.columns:
        msg = "ETPs not determined, but required for curve fit evaluaton."
        raise ValueError(msg)
    outputs = qa.evaluate_bell_curve_error(
        peak_idxs=peak_set.peak_df["peak_idx"].to_numpy(),
        start_idxs=peak_set.peak_df["start_idx"].to_numpy(),
        end_idxs=peak_set.peak_df["end_idx"].to_numpy(),
        signal=peak_set.signal,
        fs=timeseries.param["fs"],
        time_products=peak_set.peak_df[parameter_names["time_product"]].to_numpy(),
        bell_window_s=cutoff["bell_window_s"],
        bell_threshold=cutoff["curve_fit"],
    )
    (valid_bell_shape, _, percentage_bell_error, y_min, parameters) = outputs

    peak_set.peak_df["bell_y_min"] = y_min
    for idx, parameter in enumerate(["a", "b", "c"]):
        peak_set.peak_df["bell_" + parameter] = parameters[:, idx]

    quality_values_df["bell"] = percentage_bell_error
    quality_outcomes_df["bell"] = valid_bell_shape
    return quality_outcomes_df, quality_values_df, peak_set


def test_relative_aub(
    peak_set: PeaksSet, quality_outcomes_df: pd.DataFrame, cutoff: dict[str, float]
) -> pd.DataFrame:
    """Test the relative area under the baseline.

    See TimeSeries.test_emg_quality method in resurfemg.data_connector.data_classes
    for more information.

    Args:
      peak_set (PeaksSet):
      quality_outcomes_df (pd.DataFrame):
      cutoff (dict[str, float]):

    Returns:

    """
    if "AUB" not in peak_set.peak_df.columns:
        msg = (
            "AUB not determined, but required for relative area "
            "under the baseline (AUB) evaluaton."
        )
        raise ValueError(msg)
    valid_relative_aubs = qa.detect_local_high_aub(
        aubs=peak_set.peak_df["AUB"].to_numpy(),
        threshold_percentile=cutoff["relative_aub_percentile"],
        threshold_factor=cutoff["relative_aub_factor"],
    )
    quality_outcomes_df["relative_aub"] = valid_relative_aubs
    return quality_outcomes_df


def test_relative_etp(
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    cutoff: dict[str, float],
    parameter_names: dict[str, str],
) -> pd.DataFrame:
    """Evaluate extremely low and high timeproducts.

    See TimeSeries.test_emg_quality method in resurfemg.data_connector.data_classes
    for more information.

    Args:
      peak_set (PeaksSet):
      quality_outcomes_df (pd.DataFrame):
      cutoff (dict[str, float]):
      parameter_names (dict[str, str]):

    Returns:

    """
    if parameter_names["time_product"] not in peak_set.peak_df.columns:
        msg = "ETPs not determined, but required for curve fit evaluaton."
        raise ValueError(msg)
    valid_etps = qa.detect_extreme_time_products(
        time_products=peak_set.peak_df[parameter_names["time_product"]].to_numpy(),
        upper_percentile=cutoff["relative_etp_upper_percentile"],
        upper_factor=cutoff["relative_etp_upper_factor"],
        lower_percentile=cutoff["relative_etp_lower_percentile"],
        lower_factor=cutoff["relative_etp_lower_factor"],
    )
    quality_outcomes_df["relative_etp"] = valid_etps
    return quality_outcomes_df


def initialize_pocc_tests(
    timeseries: TimeSeries,
    peak_set_name: str,
    cutoff: dict[str, float] | Literal["tolerant", "strict"] | None,
    skip_tests: list[str],
    parameter_names: dict[str, str],
) -> tuple[
    list[str],
    dict[str, float],
    PeaksSet,
    dict[str, str],
    int,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Initialize local parameters.

    See TimeSeries.test_pocc_quality method in
    resurfemg.data_connector.data_classes for more information.

    Args:
      timeseries (TimeSeries):
      peak_set_name (str):
      cutoff (dict[str, float] | Literal["tolerant", "strict"] | None):
      skip_tests (list[str]):
      parameter_names (dict[str, str]):

    Returns:

    """
    if peak_set_name in timeseries.peaks:
        peak_set = timeseries.peaks[peak_set_name]
    else:
        msg = "Non-existent PeaksSet key"
        raise KeyError(msg)

    if skip_tests is None:
        skip_tests = []

    if (
        cutoff is None
        or (isinstance(cutoff, str) and cutoff == "tolerant")
        or (isinstance(cutoff, str) and cutoff == "strict")
    ):
        cutoff = {}
        cutoff["consecutive_poccs"] = 0
        cutoff["dP_up_10"] = 0.0
        cutoff["dP_up_90"] = 2.0
        cutoff["dP_up_90_norm"] = 0.8
    elif isinstance(cutoff, dict):
        tests = ["consecutive_poccs", "pocc_upslope"]
        tests_crit = {}
        tests_crit["consecutive_poccs"] = ["consecutive_poccs"]
        tests_crit["pocc_upslope"] = ["dP_up_10", "dP_up_90", "dP_up_90_norm"]

        for _, test in enumerate(tests):
            if test not in skip_tests:
                for test_crit in tests_crit[test]:
                    if test_crit not in cutoff:
                        raise KeyError("No cut-off value provided for: " + test)

    if parameter_names is None:
        parameter_names = {}

    for parameter in ["ventilator_breaths", "time_product", "AUB"]:
        if parameter not in parameter_names:
            parameter_names[parameter] = parameter

    n_peaks = len(peak_set.peak_df["peak_idx"].to_numpy())
    quality_values_df = pd.DataFrame(
        data=peak_set.peak_df["peak_idx"], columns=pd.Index(["peak_idx"])
    )
    quality_outcomes_df = pd.DataFrame(
        data=peak_set.peak_df["peak_idx"], columns=pd.Index(["peak_idx"])
    )
    return (
        skip_tests,
        cutoff,
        peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df,
    )


def test_consecutive_poccs(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    parameter_names: dict[str, str],
) -> pd.DataFrame:
    """Test for consecutive Pocc manoeuvres.

    See TimeSeries.test_pocc_quality
    method in resurfemg.data_connector.data_classes for more information.

    Args:
      timeseries (TimeSeries):
      peak_set (PeaksSet):
      quality_outcomes_df (pd.DataFrame):
      parameter_names (dict[str, str]):

    Returns:

    """
    if parameter_names["ventilator_breaths"] not in timeseries.peaks:
        msg = (
            "Ventilator breaths not determined, but required for "
            "consecutive Pocc evaluation."
        )
        raise ValueError(msg)
    vent_breaths = parameter_names["ventilator_breaths"]
    ventilator_breath_idxs = (
        timeseries.peaks[vent_breaths].peak_df["peak_idx"].to_numpy()
    )
    valid_manoeuvres = qa.detect_non_consecutive_manoeuvres(
        ventilator_breath_idxs=ventilator_breath_idxs,
        manoeuvres_idxs=peak_set.peak_df["peak_idx"].to_numpy(),
    )
    quality_outcomes_df["consecutive_poccs"] = valid_manoeuvres
    return quality_outcomes_df


def test_pocc_upslope(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    cutoff: dict[str, float],
    parameter_names: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test for sudden Pocc release.

    See TimeSeries.test_pocc_quality method in
    resurfemg.data_connector.data_classes for more information.

    Args:
      timeseries (TimeSeries):
      peak_set (PeaksSet):
      quality_outcomes_df (pd.DataFrame):
      quality_values_df (pd.DataFrame):
      cutoff (dict[str, float]):
      parameter_names (dict[str, str]):

    Returns:

    """
    if "end_idx" not in peak_set.peak_df.columns:
        msg = "Pocc end_idx not determined, but required for Pocc upslope evaluaton."
        raise ValueError(msg)
    if parameter_names["time_product"] not in peak_set.peak_df.columns:
        msg = "PTPs not determined, but required for Pocc upslope evaluaton."
        raise ValueError(msg)
    valid_poccs, criteria_matrix = qa.pocc_quality(
        p_vent_signal=timeseries["raw"],
        pocc_peaks=peak_set.peak_df["peak_idx"].to_numpy(),
        pocc_ends=peak_set.peak_df["end_idx"].to_numpy(),
        ptp_occs=peak_set.peak_df[parameter_names["time_product"]].to_numpy(),
        dp_up_10_threshold=cutoff["dP_up_10"],
        dp_up_90_threshold=cutoff["dP_up_90"],
        dp_up_90_norm_threshold=cutoff["dP_up_90_norm"],
    )
    quality_values_df["dP_up_10"] = criteria_matrix[0, :]
    quality_values_df["dP_up_90"] = criteria_matrix[1, :]
    quality_values_df["dP_up_90_norm"] = criteria_matrix[2, :]

    quality_outcomes_df["pocc_upslope"] = valid_poccs
    return quality_outcomes_df, quality_values_df


def initialize_linked_peaks_tests(
    timeseries: TimeSeries,
    peak_set_name: str,
    linked_timeseries: TimeSeries,
    linked_peak_set_name: str,
    cutoff: dict[str, float] | Literal["tolerant", "strict"] | None,
    skip_tests: list[str] | None,
    parameter_names: dict[str, str] | None,
) -> tuple[
    list[str],
    dict[str, float],
    PeaksSet,
    PeaksSet,
    dict[str, str],
    int,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Initialize local parameter.

    See TimeSeries.test_linked_peak_sets method
    in resurfemg.data_connector.data_classes for more information.

    Args:
      timeseries (TimeSeries):
      peak_set_name (str):
      linked_timeseries (TimeSeries):
      linked_peak_set_name (str):
      cutoff (dict[str, float] | Literal["tolerant", "strict"] | None):
      skip_tests (list[str] | None):
      parameter_names (dict[str, str] | None):

    Returns:

    """
    if peak_set_name in timeseries.peaks:
        peak_set = timeseries.peaks[peak_set_name]
    else:
        msg = "Non-existent PeaksSet key"
        raise KeyError(msg)

    if linked_peak_set_name in linked_timeseries.peaks:
        linked_peak_set = linked_timeseries.peaks[linked_peak_set_name]
    else:
        msg = "Non-existent linked PeaksSet key"
        raise KeyError(msg)

    if skip_tests is None:
        skip_tests = []

    if (cutoff is None) or (isinstance(cutoff, str) and cutoff == "tolerant"):
        cutoff = {}
        cutoff["fraction_emg_breaths"] = 0.1
        cutoff["delta_min"] = -0.5
        cutoff["delta_max"] = 2.0
    elif isinstance(cutoff, str) and cutoff == "strict":
        cutoff = {}
        cutoff["fraction_emg_breaths"] = 0.5
        cutoff["delta_min"] = -0.5
        cutoff["delta_max"] = 2.0
    elif isinstance(cutoff, dict):
        tests = ["fraction_emg_breaths", "event_timing"]
        tests_crit = {}
        tests_crit["fraction_emg_breaths"] = ["fraction_emg_breaths"]
        tests_crit["event_timing"] = ["delta_min", "delta_max"]

        for _, test in enumerate(tests):
            if test not in skip_tests:
                for crit in tests_crit[test]:
                    if crit not in cutoff:
                        raise KeyError("No cut-off value provided for: " + test)

    if parameter_names is None:
        parameter_names = {}

    for parameter in ["ventilator_breaths", "rr"]:
        if parameter not in parameter_names:
            parameter_names[parameter] = parameter

    n_peaks = len(peak_set.peak_df["peak_idx"].to_numpy())
    quality_values_df = pd.DataFrame(
        data=peak_set.peak_df["peak_idx"], columns=pd.Index(["peak_idx"])
    )
    quality_outcomes_df = pd.DataFrame(
        data=peak_set.peak_df["peak_idx"], columns=pd.Index(["peak_idx"])
    )
    return (
        skip_tests,
        cutoff,
        peak_set,
        linked_peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df,
    )


def test_fraction_detected_breaths(
    native_peak_set: PeaksSet,
    linked_time_series: TimeSeries,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    n_peaks: int,
    cutoff: dict[str, float],
    parameter_names: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test detected peak fraction.

    See TimeSeries.test_linked_peak_sets method
    in resurfemg.data_connector.data_classes for more information.

    Args:
      native_peak_set (PeaksSet):
      linked_time_series (TimeSeries):
      quality_outcomes_df (pd.DataFrame):
      quality_values_df (pd.DataFrame):
      n_peaks (int):
      cutoff: dict[str:
      float]:
      parameter_names: dict[str:
      str]:

    Returns:

    """
    native_peak_idxs = native_peak_set.peak_df["peak_idx"].to_numpy()

    fraction_emg_breaths, _ = qa.evaluate_respiratory_rates(
        emg_breath_idxs=native_peak_idxs,
        t_emg=max(native_peak_set.t_data),
        rr_vent=linked_time_series.param[parameter_names["rr"]],
    )
    valid_fraction = fraction_emg_breaths > cutoff["fraction_emg_breaths"]
    fraction_emg_breaths_vec = np.array(n_peaks * [fraction_emg_breaths])
    valid_fraction_vec = np.array(n_peaks * [valid_fraction])
    quality_outcomes_df["detected_fraction"] = valid_fraction_vec
    quality_values_df["detected_fraction"] = fraction_emg_breaths_vec
    return quality_outcomes_df, quality_values_df


def test_event_timing(
    timeseries: TimeSeries,
    native_peak_set: PeaksSet,
    linked_timeseries: TimeSeries,
    linked_peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    cutoff: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test event timing relative to other peak_set.

    See TimeSeries.test_linked_peak_sets method in
    resurfemg.data_connector.data_classes for more information.

    Args:
      timeseries (TimeSeries):
      native_peak_set (PeaksSet):
      linked_timeseries (TimeSeries):
      linked_peak_set (PeaksSet):
      quality_outcomes_df: pd.DataFrame:
      quality_values_df: pd.DataFrame:
      cutoff: dict[str:
      float]:

    Returns:

    """
    native_peak_idxs = native_peak_set.peak_df["peak_idx"].to_numpy()
    linked_peak_idxs = linked_peak_set.peak_df["peak_idx"].to_numpy()

    t_events_1 = native_peak_idxs / timeseries.param["fs"]
    t_events_2 = linked_peak_idxs / linked_timeseries.param["fs"]

    correct_timing, delta_time = qa.evaluate_event_timing(
        t_events_1,
        t_events_2,
        delta_min=cutoff["delta_min"],
        delta_max=cutoff["delta_max"],
    )
    quality_outcomes_df["event_timing"] = correct_timing
    quality_values_df["event_delta_time"] = delta_time
    return quality_outcomes_df, quality_values_df


def test_emg_quality(
    timeseries: TimeSeries,
    peak_set_name: str,
    cutoff: dict[str, float] | None = None,
    skip_tests: list[str] | None = None,
    parameter_names: dict[str, str] | None = None,
    verbose: bool = True,
) -> None:
    """Test EMG PeakSet quality.

    Test EMG PeaksSet according to quality criteria in Warnaar et al.
    (2024): interpeak_distance, snr, aub, curve_fit, and extended with
    relative area under the baseline (relative_aub) and relative ETP
    (relative_ETP). Peak validity is updated in the PeaksSet object.
    See postprocessing.quality_assessment submodule for details.
    -----------------------------------------------------------------------

    Args:
      peak_set_name(str): PeaksSet name in timeseries.peaks dict
      cutoff(dict): Cut-off criteria for passing the tests 'tolerant' and
    'strict' can also be providedto use the values from Warnaar et al.
      skip_tests(list): List of tests to skip.
      parameter_names(dict): Optionally refer to custom parameter names for
    default PeaksSet (ecg)
      verbose(bool): Output the test values, and pass/fail to console.
      timeseries (TimeSeries):
      peak_set_name (str):
      cutoff: dict[str:
      float] | None:  (Default value = None)
      skip_tests: list[str] | None:  (Default value = None)
      parameter_names: dict[str:
      str] | None:  (Default value = None)
      verbose (bool):  (Default value = True)

    Returns:
      None: None

    """
    output = initialize_emg_tests(
        timeseries, peak_set_name, cutoff, skip_tests, parameter_names
    )
    (
        skip_tests,
        cutoff,
        peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df,
    ) = output

    if "interpeak_dist" not in skip_tests:
        quality_outcomes_df = test_interpeak_distance(
            timeseries,
            peak_set,
            quality_outcomes_df,
            n_peaks,
            cutoff,
            parameter_names,
        )

    if "snr" not in skip_tests:
        quality_outcomes_df = test_snr(
            timeseries,
            peak_set,
            quality_outcomes_df,
            quality_values_df,
            cutoff,
            parameter_names,
        )

    if "aub" not in skip_tests:
        quality_outcomes_df, quality_values_df = test_aub(
            timeseries,
            peak_set,
            quality_outcomes_df,
            quality_values_df,
            cutoff,
            parameter_names,
        )

    if "curve_fit" not in skip_tests:
        quality_outcomes_df, quality_values_df, peak_set = test_curve_fits(
            timeseries,
            peak_set,
            quality_outcomes_df,
            quality_values_df,
            cutoff,
            parameter_names,
        )

    if "relative_aub" not in skip_tests:
        quality_outcomes_df = test_relative_aub(peak_set, quality_outcomes_df, cutoff)

    if "relative_etp" not in skip_tests:
        quality_outcomes_df = test_relative_etp(
            peak_set, quality_outcomes_df, cutoff, parameter_names
        )

    peak_set.update_test_outcomes(quality_values_df)
    peak_set.evaluate_validity(quality_outcomes_df)
    if verbose:
        print("Test values:\n", peak_set.quality_values_df)
        print("Test outcomes:\n", peak_set.quality_outcomes_df)


def test_pocc_quality(
    timeseries: TimeSeries,
    peak_set_name: str,
    cutoff: dict[str, float] | Literal["tolerant", "strict"] | None = None,
    skip_tests: list[str] | None = None,
    parameter_names: dict | None = None,
    verbose: bool = True,
) -> None:
    """Test EMG PeaksSet according to quality criteria in Warnaar et al.
    (2024): consecutive_poccs, and pocc_upslope. Peak validity is updated
    in the PeaksSet object.
    -----------------------------------------------------------------------

    Args:
      peak_set_name(str): PeaksSet name in timeseries.peaks dict
      cutoff(dict): Cut-off criteria for passing the tests. 'tolerant' and
    'strict' can also be provided  to use the values from Warnaar et al.
      skip_tests(list): List of tests to skip.
      parameter_names(dict): Refer to custom parameter names for default
    PeaksSet and parameters (ventilator_breaths, time_product, AUB)
      verbose(bool): Output the test values, and pass/fail to console.
      timeseries (TimeSeries):
      peak_set_name (str):
      cutoff: dict[str:
      float] | Literal["tolerant":
      "strict"] | None:  (Default value = None)
      skip_tests: list[str] | None:  (Default value = None)
      parameter_names: dict | None:  (Default value = None)
      verbose (bool):  (Default value = True)

    Returns:
      None: None

    """
    output = initialize_pocc_tests(
        timeseries, peak_set_name, cutoff, skip_tests, parameter_names
    )
    (
        skip_tests,
        cutoff,
        peak_set,
        parameter_names,
        _,
        quality_values_df,
        quality_outcomes_df,
    ) = output

    if "consecutive_poccs" not in skip_tests:
        quality_outcomes_df = test_consecutive_poccs(
            timeseries, peak_set, quality_outcomes_df, parameter_names
        )

    if "pocc_upslope" not in skip_tests:
        quality_outcomes_df, quality_values_df = test_pocc_upslope(
            timeseries,
            peak_set,
            quality_outcomes_df,
            quality_values_df,
            cutoff,
            parameter_names,
        )

    peak_set.update_test_outcomes(quality_values_df)
    peak_set.evaluate_validity(quality_outcomes_df)
    if verbose:
        print("Test values:\n", peak_set.quality_values_df)
        print("Test outcomes:\n", peak_set.quality_outcomes_df)


def test_linked_peak_sets(
    timeseries: TimeSeries,
    peak_set_name: str,
    linked_timeseries: TimeSeries,
    linked_peak_set_name: str,
    parameter_names: dict[str, str] | None = None,
    cutoff: dict[str, float] | Literal["tolerant", "strict"] | None = None,
    skip_tests: list[str] | None = None,
    verbose: bool = True,
) -> None:
    """Test number of detected breaths in the native PeaksSet compared to
    number (fraction_emg_breaths) of and timing peaks (event_timing) in the
    linked PeaksSet. Peak validity is updated in the PeaksSet object.
    -----------------------------------------------------------------------

    Args:
      peak_set_name(str): PeaksSet name in timeseries.peaks dict
      linked_timeseries(TimeSeries): TimeSeries object with linked signal
      linked_peak_set_name(str): PeaksSet name in linked_timeseries.peaks
      parameter_names(dict): Optionally refer to custom parameter names for
    default PeaksSet and parameter names ('rr', )
      cutoff(dict): Cut-off criteria for passing the tests. 'tolerant' and
    'strict' can also be provided use the pre-defined values.
      skip_tests(list): List of tests to skip.
      verbose(bool): Output the test values, and pass/fail to console.
      timeseries (TimeSeries):
      peak_set_name (str):
      linked_timeseries (TimeSeries):
      linked_peak_set_name (str):
      parameter_names: dict[str:
      str] | None:  (Default value = None)
      cutoff: dict[str:
      float] | Literal["tolerant":
      "strict"] | None:  (Default value = None)
      skip_tests: list[str] | None:  (Default value = None)
      verbose (bool):  (Default value = True)

    Returns:
      None: None

    """
    output = initialize_linked_peaks_tests(
        timeseries,
        peak_set_name,
        linked_timeseries,
        linked_peak_set_name,
        cutoff,
        skip_tests,
        parameter_names,
    )
    (
        skip_tests,
        cutoff,
        native_peak_set,
        linked_peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df,
    ) = output

    if "fraction_emg_breaths" not in skip_tests:
        quality_outcomes_df, quality_values_df = test_fraction_detected_breaths(
            native_peak_set,
            linked_timeseries,
            quality_outcomes_df,
            quality_values_df,
            n_peaks,
            cutoff,
            parameter_names,
        )

    if "event_timing" not in skip_tests:
        quality_outcomes_df, quality_values_df = test_event_timing(
            timeseries,
            native_peak_set,
            linked_timeseries,
            linked_peak_set,
            quality_outcomes_df,
            quality_values_df,
            cutoff,
        )

    native_peak_set.update_test_outcomes(quality_values_df)
    native_peak_set.evaluate_validity(quality_outcomes_df)
    if verbose:
        print("Test values:\n", native_peak_set.quality_values_df)
        print("Test outcomes:\n", native_peak_set.quality_outcomes_df)
