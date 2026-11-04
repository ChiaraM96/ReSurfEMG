"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains the quality assessment methods for data classes.
"""  # noqa: D205

from typing import Literal

import numpy as np
import pandas as pd

from resurfemg.data_connector.data_classes import TimeSeries
from resurfemg.data_connector.peakset_class import PeaksSet
from resurfemg.postprocessing import quality_assessment as qa

CUTOFF_DICT_TYPE = dict[str, float]
CUTOFF_OPTIONS_TYPE: Literal["tolerant", "strict"]
CUTOFF_TYPES = CUTOFF_DICT_TYPE | Literal["tolerant", "strict"]
PARAMETER_NAMES_TYPE = dict[str, str]
THRESHOLD_TYPES = int | float | np.number
AUB_WINDOW_SECONDS = 5
BELLS_WINDOW_SECONDS = 5
CUTOFF_DEFAULTS = {
    "tolerant": {
        "interpeak_distance": 1.1,
        "snr": 1.4,
        "aub": 40,
        "curve_fit": 30,
        "relative_aub_percentile": 75.0,
        "relative_aub_factor": 4.0,
        "relative_etp_upper_percentile": 95.0,
        "relative_etp_upper_factor": 10.0,
        "relative_etp_lower_percentile": 5.0,
        "relative_etp_lower_factor": 0.1,
    },
    "strict": {
        "interpeak_distance": 1.1,
        "snr": 1.75,
        "aub": 30,
        "curve_fit": 25,
        "relative_aub_percentile": 75.0,
        "relative_aub_factor": 2.0,
        "relative_etp_upper_percentile": 95.0,
        "relative_etp_upper_factor": 10.0,
        "relative_etp_lower_percentile": 5.0,
        "relative_etp_lower_factor": 0.1,
    },
}

TESTS_EMG = ["interpeak_distance", "snr", "aub", "curve_fit"]
TESTS_LINKED = {
    "fraction_emg_breaths": ["fraction_emg_breaths"],
    "event_timing": ["delta_min", "delta_max"],
}


def _check_cutoff_and_skips(
    test: str | tuple[str, str],
    cutoff: CUTOFF_DICT_TYPE,
    skip_tests: list[str],
) -> None:
    if isinstance(test, tuple):
        test, crit = test
    else:
        crit = test
    if test not in skip_tests:
        if crit not in cutoff:
            raise KeyError("No cut-off value provided for: " + crit)
        if not isinstance(cutoff[crit], THRESHOLD_TYPES):
            raise ValueError("Invalid cut-off value provided for: " + crit)


def _check_aub_cutoffs(
    test: str, cutoff: CUTOFF_DICT_TYPE, skip_tests: list[str]
) -> None:
    if "aub" not in skip_tests and "aub_window_s" not in cutoff:
        raise KeyError(
            "No area under the baseline window provided for: " + test
        )
    if "relative_aub" not in skip_tests:
        if "relative_aub_percentile" not in cutoff:
            msg_0 = (
                "No relative area under the baseline percentile"
                " provided for: relative_aub"
            )
            raise KeyError(msg_0)
        if "relative_aub_factor" not in cutoff:
            msg_0 = (
                "No relative area under the baseline factor provided for:"
                " relative_aub"
            )
            raise KeyError(msg_0)


def _check_etp_cutoffs(
    cutoff: CUTOFF_DICT_TYPE, skip_tests: list[str]
) -> None:
    if "relative_etp" not in skip_tests:
        for _param in [
            "upper_percentile",
            "upper_factor",
            "lower_percentile",
            "lower_factor",
        ]:
            if ("relative_etp_" + _param) not in cutoff:
                msg_0 = f"No relative_etp_{_param} provided for: relative_etp"
                raise KeyError(msg_0)


def initialize_emg_tests(
    timeseries: TimeSeries,
    peak_set_name: str,
    cutoff: CUTOFF_TYPES = "tolerant",
    skip_tests: list[str] | None = None,
    parameter_names: PARAMETER_NAMES_TYPE | None = None,
) -> tuple[
    list[str],
    CUTOFF_DICT_TYPE,
    PeaksSet,
    PARAMETER_NAMES_TYPE,
    int,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Initialize local parameters.

    See TimeSeries.test_emg_quality method in
    resurfemg.data_connector.data_classes for more information.

    Parameters
    ----------
    timeseries : TimeSeries
        TimeSeries object containing the EMG data and associated PeaksSet
        to test
    peak_set_name : str
        Name of the PeaksSet in timeseries.peaks dict to test
    cutoff : str or dict, optional
        The cut-off criteria for passing the tests, passed either as a string
        indicating the default criteria to use ("tolerant" or "strict"),
        or as a dictionary providing the cut-off values for each test.
    skip_tests : list, optional
        A list of tests to skip. Example: `skip_tests = ['snr', 'aub']`
    parameter_names : dict, optional
        A dictionary specifying the namesby which the default parameters are
        referred to in the PeaksSet.

    Returns
    -------
    skip_tests : list
        List of skipped tests.
    cutoff : dict
        Cut-off criteria for passing the tests.
    peak_set : PeaksSet
        PeaksSet object for the native peaks.
    parameter_names : dict
        A dictionary specifying the namesby which the default parameters are
        referred to in the PeaksSet.
    n_peaks : int
        Number of peaks in the native PeaksSet.
    quality_values_df : pd.DataFrame
        DataFrame to store quality values.
    quality_outcomes_df : pd.DataFrame
        DataFrame to store quality outcomes.

    Raises
    ------
    KeyError
        If the specified peak_set_name does not exist in the TimeSeries object,
        or if required cutoff values are not provided in the cutoff dictionary.
    ValueError
        If the provided cutoff value is not a float.

    Examples
    --------
        >>> (skipped_tests, cutoff, peak_set, parameter_names, n_peaks,
        ... quality_values_df, quality_outcomes_df) = initialize_emg_tests(
        ...                 timeseries=timeseries,
        ...                 peak_set_name='my_peak_set',
        ...                 parameter_names = {'ecg': 'ECG',
        ...                             'time_product': 'PTP',
        ...                             'baseline': 'baseline'},
        ...                 cutoff = {'interpeak_distance': 1.1,
        ...                     'snr': 1.4,
        ...                     'aub': 40,
        ...                     'curve_fit': 30,},
        ...                 skip_tests = ['snr', 'aub']
        ...             )

    """
    if peak_set_name in timeseries.peaks:
        peak_set = timeseries.peaks[peak_set_name]
    else:
        msg = "Non-existent PeaksSet key"
        raise KeyError(msg)
    skip_tests = skip_tests if skip_tests is not None else []
    cutoff = cutoff if cutoff is not None else "tolerant"
    parameter_names = parameter_names if parameter_names is not None else {}
    match cutoff:
        case "tolerant" | "strict":
            cutoff = {
                **CUTOFF_DEFAULTS[cutoff],
                "aub_window_s": 5 * timeseries.param["fs"],
                "bell_window_s": 5 * timeseries.param["fs"],
            }
        case dict(cutoff):
            for test in TESTS_EMG:
                _check_cutoff_and_skips(test, cutoff, skip_tests)
                _check_aub_cutoffs(test, cutoff, skip_tests)
                _check_etp_cutoffs(cutoff, skip_tests)
        case _:
            msg_0 = "Invalid cut-off value provided."
            raise ValueError(msg_0)

    for parameter in ["ecg", "time_product", "baseline"]:
        if parameter not in parameter_names:
            parameter_names[parameter] = parameter

    n_peaks = len(peak_set.peak_df["peak_idx"].to_numpy())
    quality_values_df = pd.DataFrame(
        data=peak_set.peak_df["peak_idx"], columns=pd.Index(["peak_idx"])
    )
    quality_outcomes_df = quality_values_df.copy()
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
    cutoff: CUTOFF_DICT_TYPE,
    parameter_names: PARAMETER_NAMES_TYPE,
) -> pd.DataFrame:
    """Test interpeak distance.

    See TimeSeries.test_emg_quality method in
    resurfemg.data_connector.data_classes for more information.

    Parameters
    ----------
    timeseries : TimeSeries
        TimeSeries object containing the EMG data and associated PeaksSet
        to test
    peak_set : PeaksSet
        The PeaksSet to test
    quality_outcomes_df : pd.DataFrame
        DataFrame containing the outcomes of the quality tests
    n_peaks : int
        Number of peaks in the native PeaksSet.
    cutoff : dict
        Cut-off criteria for passing the tests.
    parameter_names : dict
        A dictionary specifying the namesby which the default parameters are
        referred to in the PeaksSet.

    Returns
    -------
    quality_outcomes_df : pd.DataFrame
        The updated DataFrame containing the outcomes of the quality tests
        after evaluating the interpeak distance.

    Raises
    ------
    ValueError
        If ECG peaks are not determined in the TimeSeries object, which are
        required for interpeak distance evaluation.

    """
    if parameter_names["ecg"] not in timeseries.peaks:
        msg = (
            "ECG peaks not determined, but required for interpeak"
            " distance evaluation."
        )
        raise ValueError(msg)
    ecg_peaks = (
        timeseries.peaks[parameter_names["ecg"]].peak_df["peak_idx"].to_numpy()
    )
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
    cutoff: CUTOFF_DICT_TYPE,
    parameter_names: PARAMETER_NAMES_TYPE,
) -> pd.DataFrame:
    """Test signal-to-noise ratio.

        See TimeSeries.test_emg_quality method in
        resurfemg.data_connector.data_classes for more information.

    Parameters
    ----------
    timeseries : TimeSeries
        TimeSeries object containing the EMG data and associated PeaksSet
        to test
    peak_set : PeaksSet
        The PeaksSet to test
    quality_outcomes_df : pd.DataFrame
        DataFrame containing the outcomes of the quality tests
    quality_values_df : pd.DataFrame
        DataFrame containing the values of the metrics used in the tests
    cutoff : dict
        Cut-off criteria for passing the tests.
    parameter_names : dict
        A dictionary specifying the namesby which the default parameters are
        referred to in the PeaksSet.

    Returns
    -------
    quality_outcomes_df : pd.DataFrame
        The updated DataFrame containing the outcomes of the quality tests
        after evaluating the signal-to-noise ratio.

    Raises
    ------
    ValueError
        If the baseline is not determined in the TimeSeries object, which is
        required for SNR evaluation.


    """
    if parameter_names["baseline"] not in timeseries._y_data:  # noqa: SLF001
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
    cutoff: CUTOFF_DICT_TYPE,
    parameter_names: PARAMETER_NAMES_TYPE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test percentage area under the baseline.

    See TimeSeries.test_emg_quality method in
    `resurfemg.data_connector.data_classes` for more information.

    Parameters
    ----------
    timeseries : TimeSeries
        TimeSeries object containing the EMG data and associated PeaksSet
        to test
    peak_set : PeaksSet
        The PeaksSet to test
    quality_outcomes_df : pd.DataFrame
        DataFrame containing the outcomes of the quality tests
    quality_values_df : pd.DataFrame
        DataFrame containing the values of the metrics used in the tests
    cutoff : dict
        Cut-off criteria for passing the tests.
    parameter_names : dict
        A dictionary specifying the namesby which the default parameters are
        referred to in the PeaksSet.

    Returns
    -------
    quality_outcomes_df : pd.DataFrame
        The updated DataFrame containing the outcomes of the quality tests
        after evaluating the area under the baseline.
    quality_values_df : pd.DataFrame
        The updated DataFrame containing the values of the metrics used in the
        quality tests.

    Raises
    ------
    ValueError
        - If the baseline is not determined in the TimeSeries object
        - If the start_idxs are not determined in the PeaksSet
        - If the end_idxs are not determined in the PeaksSet

    """
    if parameter_names["baseline"] not in timeseries._y_data:  # noqa: SLF001
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
    cutoff: CUTOFF_DICT_TYPE,
    parameter_names: PARAMETER_NAMES_TYPE,
) -> tuple[pd.DataFrame, pd.DataFrame, PeaksSet]:
    """Test curve fit.

    See TimeSeries.test_emg_quality method in
    resurfemg.data_connector.data_classes for more information.

    Parameters
    ----------
    timeseries : TimeSeries
        TimeSeries object containing the EMG data and associated PeaksSet
        to test
    peak_set : PeaksSet
        The PeaksSet to test
    quality_outcomes_df : pd.DataFrame
        DataFrame containing the outcomes of the quality tests
    quality_values_df : pd.DataFrame
        DataFrame containing the values of the metrics used in the tests
    cutoff : dict
        Cut-off criteria for passing the tests.
    parameter_names : dict
        A dictionary specifying the namesby which the default parameters are
        referred to in the PeaksSet.

    Returns
    -------
    quality_outcomes_df : pd.DataFrame
        The updated DataFrame containing the outcomes of the quality tests
        after evaluating the curve fit.
    quality_values_df : pd.DataFrame
        The updated DataFrame containing the values of the metrics used in the
        quality tests.
    peak_set : PeaksSet
        The updated PeaksSet object

    Raises
    ------
    ValueError
        - If the baseline is not determined in the TimeSeries object
        - If the start_idxs or end_idxs are not determined in the PeaksSet
        - If the time products are not determined in the PeaksSet

    """
    if (
        parameter_names["baseline"] not in timeseries._y_data  # noqa: SLF001
        or timeseries[parameter_names["baseline"]] is None
    ):
        msg = (
            "Baseline not determined, but required for "
            " area under the baseline (AUB) evaluaton."
        )
        raise ValueError(msg)

    if "start_idx" not in peak_set.peak_df.columns:
        msg = (
            "start_idxs not determined, but required for  curve fit evaluaton."
        )
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
        time_products=peak_set.peak_df[
            parameter_names["time_product"]
        ].to_numpy(),
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
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    cutoff: CUTOFF_DICT_TYPE,
) -> pd.DataFrame:
    """Test the relative area under the baseline.

    See `TimeSeries.test_emg_quality method` in
    `resurfemg.data_connector.data_classes`
    for more information.

    Parameters
    ----------
    peak_set : PeaksSet
        The PeaksSet to test
    quality_outcomes_df : pd.DataFrame
        DataFrame containing the outcomes of the quality tests
    cutoff : dict
        Cut-off criteria for passing the tests.

    Returns
    -------
    quality_outcomes_df : pd.DataFrame
        The updated DataFrame containing the outcomes of the quality tests
        after evaluating the relative area under the baseline.

    Raises
    ------
    ValueError
         If the AUB is not determined in the PeaksSet

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
    cutoff: CUTOFF_DICT_TYPE,
    parameter_names: PARAMETER_NAMES_TYPE,
) -> pd.DataFrame:
    """Evaluate extremely low and high timeproducts.

    See `TimeSeries.test_emg_quality` method in
    `resurfemg.data_connector.data_classes` for more information.

    Parameters
    ----------
    peak_set : PeaksSet
        The PeaksSet to test
    quality_outcomes_df : pd.DataFrame
        DataFrame containing the outcomes of the quality tests
    cutoff : dict
        Cut-off criteria for passing the tests.
    parameter_names : dict
        A dictionary specifying the names by which the default parameters are
        referred to in the PeaksSet.

    Returns
    -------
    quality_outcomes_df : pd.DataFrame
        The updated DataFrame containing the outcomes of the quality tests
        after evaluating the relative time products.

    Raises
    ------
    ValueError
        If the time products are not determined in the PeaksSet

    """
    if parameter_names["time_product"] not in peak_set.peak_df.columns:
        msg = "ETPs not determined, but required for curve fit evaluaton."
        raise ValueError(msg)
    valid_etps = qa.detect_extreme_time_products(
        time_products=peak_set.peak_df[
            parameter_names["time_product"]
        ].to_numpy(),
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
    cutoff: CUTOFF_TYPES | None,
    skip_tests: list[str] | None,
    parameter_names: PARAMETER_NAMES_TYPE | None,
) -> tuple[
    list[str],
    CUTOFF_DICT_TYPE,
    PeaksSet,
    PARAMETER_NAMES_TYPE,
    int,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Initialize local parameters.

    See `TimeSeries.test_pocc_quality` method in
    `resurfemg.data_connector.data_classes` for more information.

    Parameters
    ----------
    timeseries : TimeSeries
        The TimeSeries object containing the data to be tested
    peak_set_name : str
        The name of the PeaksSet to test
    cutoff : dict or str: {"tolerant", "strict"}, optional
        Cut-off criteria for passing the tests. Can be provided as a string
        indicating the default criteria to use ("tolerant" or "strict"),
        or as a dictionary providing the cut-off values for each test.
    skip_tests : list of str, optional
        The names of the tests to skip.
    parameter_names : dict, optional
        A dictionary specifying the names by which the default parameters are
        referred to in the PeaksSet.

    Returns
    -------
    skip_tests : list
        List of skipped tests.
    cutoff : dict
        Cut-off criteria for passing the tests.
    peak_set : PeaksSet
        The PeaksSet to test.
    parameter_names : dict
        A dictionary specifying the names by which the default parameters are
        referred to in the PeaksSet.
    n_peaks : int
        The number of peaks in the PeaksSet.
    quality_values_df : pd.DataFrame
        DataFrame containing the values of the quality tests.
    quality_outcomes_df : pd.DataFrame
        DataFrame containing the outcomes of the quality tests.

    Raises
    ------
    KeyError
        - If the specified peak_set_name does not exist in the TimeSeries.
        - If required cut-off values are not provided in the cutoff dictionary.

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
                        raise KeyError(
                            "No cut-off value provided for: " + test
                        )

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
    parameter_names: PARAMETER_NAMES_TYPE,
) -> pd.DataFrame:
    """Test for consecutive Pocc manoeuvres.

    See `TimeSeries.test_pocc_quality` method in
    `resurfemg.data_connector.data_classes` for more information.

    Parameters
    ----------
    timeseries : TimeSeries
        TimeSeries object containing the ventilator data
    peak_set : PeaksSet
        The PeaksSet to test
    quality_outcomes_df : pd.DataFrame
        DataFrame containing the outcomes of the quality tests
    parameter_names : dict
        A dictionary specifying the names by which the default parameters are
        referred to in the PeaksSet.

    Returns
    -------
    quality_outcomes_df : pd.DataFrame
        The updated DataFrame containing the outcomes of the quality tests
        after evaluating the consecutive Pocc manoeuvres.

    Raises
    ------
    ValueError
        If the ventilator breaths are not determined in the TimeSeries object,
        which are required for consecutive Pocc evaluation.
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
    cutoff: CUTOFF_DICT_TYPE,
    parameter_names: PARAMETER_NAMES_TYPE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test for sudden Pocc release.

    See `TimeSeries.test_pocc_quality` method in
    `resurfemg.data_connector.data_classes` for more information.

    Parameters
    ----------
    timeseries : TimeSeries
        TimeSeries object containing the ventilator data
    peak_set : PeaksSet
        The PeaksSet to test
    quality_outcomes_df : pd.DataFrame
        DataFrame containing the outcomes of the quality tests
    quality_values_df : pd.DataFrame
        DataFrame containing the values of the metrics used in the tests
    cutoff : dict
        Cut-off criteria for passing the tests.
    parameter_names : dict
        A dictionary specifying the names by which the default parameters are
        referred to in the PeaksSet.

    Returns
    -------
    quality_outcomes_df : pd.DataFrame
        The updated DataFrame containing the outcomes of the quality tests
        after evaluating the upslope of the Pocc manoeuvres.
    quality_values_df : pd.DataFrame
        The DataFrame containing the values of the metrics used in the
        quality tests, updated with the Pocc upslope values.

    Raises
    ------
    ValueError
        If the end_idxs or time products are not determined in the PeaksSet.

    """
    if "end_idx" not in peak_set.peak_df.columns:
        msg = (
            "Pocc end_idx not determined, but required for Pocc "
            "upslope evaluaton."
        )
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
    cutoff: CUTOFF_TYPES = "tolerant",
    skip_tests: list[str] | None = None,
    parameter_names: PARAMETER_NAMES_TYPE | None = None,
) -> tuple[
    list[str],
    CUTOFF_DICT_TYPE,
    PeaksSet,
    PeaksSet,
    PARAMETER_NAMES_TYPE,
    int,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Initialize local parameter.

    See `TimeSeries.test_linked_peak_sets` method
    in `resurfemg.data_connector.data_classes` for more information.

    Parameters
    ----------
    timeseries : TimeSeries
        the TimeSeries object with with the peaks set to be tested
    peak_set_name : str
        PeaksSet name in timeseries.peaks dict
    linked_timeseries : TimeSeries
        the TimeSeries object with with the
        linked peaks set to be tested
    linked_peak_set_name : str
        PeaksSet name in linked_timeseries.peaks dict
    cutoff : dict or str
        Cut-off criteria for passing the tests 'tolerant' and
        'strict' can also be provided to use the values from Warnaar et al.
    skip_tests : list
        List of tests to skip.
    parameter_names : dict
        Optionally refer to custom parameter names for
        default PeaksSet


    Returns
    -------
    skip_tests : list
        List of skipped tests.
    cutoff: dict
        Cut-off criteria for passing the tests.
    peak_set : PeaksSet
        PeaksSet object for the native peaks.
    linked_peak_set : PeaksSet
        PeaksSet object for the linked peaks.
    parameter_names : dict
        Custom parameter names for default PeaksSet.
    n_peaks : int
        Number of peaks in the native PeaksSet.
    quality_values_df : pd.DataFrame
        DataFrame to store quality values.
    quality_outcomes_df : pd.DataFrame
        DataFrame to store quality outcomes.

    Raises
    ------
    KeyError
        - If the specified peak_set_name or linked_peak_set_name does
            not exist in the respective TimeSeries objects.
        - If required cut-off values are not provided in the cutoff dictionary.

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

    skip_tests = skip_tests if skip_tests is not None else []
    cutoff = cutoff if cutoff is not None else "tolerant"
    parameter_names = parameter_names if parameter_names is not None else {}
    match cutoff:
        case "tolerant" | "strict":
            cutoff = {
                "fraction_emg_breaths": 0.1 if cutoff == "tolerant" else 0.5,
                "delta_min": -0.5,
                "delta_max": 2.0,
            }
        case dict(cutoff):
            for test, crits in TESTS_LINKED.items():
                for crit in crits:
                    _check_cutoff_and_skips((test, crit), cutoff, skip_tests)
        case _:
            msg_0 = "Invalid cut-off value provided."
            raise ValueError(msg_0)

    for parameter in ["ventilator_breaths", "rr"]:
        if parameter not in parameter_names:
            parameter_names[parameter] = parameter

    n_peaks = len(peak_set.peak_df["peak_idx"].to_numpy())
    quality_values_df = pd.DataFrame(
        data=peak_set.peak_df["peak_idx"], columns=pd.Index(["peak_idx"])
    )
    quality_outcomes_df = quality_values_df.copy()
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
    cutoff: CUTOFF_DICT_TYPE,
    parameter_names: PARAMETER_NAMES_TYPE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test detected peak fraction.

    See TimeSeries.test_linked_peak_sets method in
    `resurfemg.data_connector.data_classes` for more information.

    Parameters
    ----------
    native_peak_set : PeaksSet
        PeaksSet object for the native peaks.
    linked_time_series : TimeSeries
        TimeSeries object for the linked time series.
    quality_outcomes_df : pd.DataFrame
        DataFrame to store quality outcomes.
    quality_values_df : pd.DataFrame
        DataFrame to store quality values.
    n_peaks : int
        Number of peaks in the native PeaksSet.
    cutoff : CUTOFF_DICT_TYPE
        Cut-off criteria for passing the tests.
    parameter_names : PARAMETER_NAMES_TYPE

    Returns
    -------
    quality_outcomes_df : pd.DataFrame
    quality_values_df : pd.DataFrame
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
    cutoff: CUTOFF_DICT_TYPE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test event timing relative to other peak_set.

    See TimeSeries.test_linked_peak_sets method in
    resurfemg.data_connector.data_classes for more information.

    Parameters
    ----------
    timeseries : TimeSeries
        TimeSeries object containing the original data
    nativepeak_set : PeaksSet
        The original PeaksSet against which the linked PeaksSet is compared
    linked_timeseries : TimeSeries
        TimeSeries object containing the linked data
    linked_peak_set : PeaksSet
        The linked PeaksSet to be compared against the native PeaksSet
    quality_outcomes_df : pd.DataFrame
        DataFrame containing the outcomes of the quality tests
    quality_values_df : pd.DataFrame
        DataFrame containing the values of the metrics used in the tests
    cutoff : dict
        Cut-off criteria for passing the tests.

    Returns
    -------
    quality_outcomes_df : pd.DataFrame
        The updated DataFrame containing the outcomes of the quality tests
        after evaluating the event timing.
    quality_values_df : pd.DataFrame
        The updated DataFrame containing the values of the metrics used in the
        quality tests after evaluating the event timing.

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
    cutoff: CUTOFF_TYPES = "tolerant",
    skip_tests: list[str] | None = None,
    parameter_names: PARAMETER_NAMES_TYPE | None = None,
    verbose: bool = True,
) -> None:
    """Test the quality of detected EMG peaks.

    Test EMG PeaksSet according to quality criteria in Warnaar et al.
    (2024): interpeak_distance, snr, aub, curve_fit, and extended with
    relative area under the baseline (relative_aub) and relative ETP
    (relative_ETP). Peak validity is updated in the PeaksSet object.
    See postprocessing.quality_assessment submodule for details.

    Parameters
    ----------
    timeseries : TimeSeries
        The TimeSeries object containing the data to be tested
    peak_set_name : str
        The name of the PeaksSet to test
    cutoff : dict, str: {"tolerant", "strict"}, optional
        Cut-off criteria for passing the tests.
        If a string is provided, it should be either "tolerant" or "strict",
        indicating the default criteria to use.
        If a dictionary is provided, it should contain the cut-off values
        for each test.
    skip_tests : list of str, optional
        The names of the tests to skip.
    parameter_names : dict, optional
        A dictionary specifying the names by which the default parameters are
        referred to in the PeaksSet.
    verbose : bool, default True
        If True, print the test values and outcomes to the console.

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

    if "interpeak_distance" not in skip_tests:
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
        quality_outcomes_df = test_relative_aub(
            peak_set, quality_outcomes_df, cutoff
        )

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
    cutoff: CUTOFF_TYPES = "tolerant",
    skip_tests: list[str] | None = None,
    parameter_names: PARAMETER_NAMES_TYPE | None = None,
    verbose: bool = True,
) -> None:
    """Test the quality of detected occluded breaths (Poccs).

    Test EMG PeaksSet according to quality criteria in Warnaar et al.
    (2024): consecutive_poccs, and pocc_upslope. Peak validity is updated
    in the PeaksSet object.


    Parameters
    ----------
    timeseries : TimeSeries
        The TimeSeries object containing the ventilator data and associated
        PeaksSet to test
    peak_set_name : str
        The name of the PeaksSet to test
    cutoff : dict, str: {"tolerant", "strict"}, optional
        Cut-off criteria for passing the tests.
        If a string is provided, it should be either "tolerant" or "strict",
        indicating the default criteria to use.
        If a dictionary is provided, it should contain the cut-off values
        for each test.
    skip_tests : list of str, optional
        The names of the tests to skip.
    parameter_names : dict, optional
        A dictionary specifying the names by which the default parameters are
        referred to in the PeaksSet.
    verbose : bool, default True
        If True, print the test values and outcomes to the console.

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
    parameter_names: PARAMETER_NAMES_TYPE | None = None,
    cutoff: CUTOFF_TYPES = "tolerant",
    skip_tests: list[str] | None = None,
    verbose: bool = True,
) -> None:
    """Test number and timing of a linked peak set against the native peak set.

    Test number of detected breaths in the native PeaksSet compared to
    number (fraction_emg_breaths) of and timing peaks (event_timing) in the
    linked PeaksSet. Peak validity is updated in the PeaksSet object.

    Parameters
    ----------
    timeseries : TimeSeries
        the TimeSeries object with with the peaks set to be tested
    peak_set_name : str
        PeaksSet name in timeseries.peaks dict
    linked_timeseries : TimeSeries
        the TimeSeries object with with the
        linked peaks set to be tested
    linked_peak_set_name : str
        PeaksSet name in linked_timeseries.peaks dict
    parameter_names : dict
        Optionally refer to custom parameter names for
        default PeaksSet
    cutoff : dict or str
        Cut-off criteria for passing the tests 'tolerant' and
        'strict' can also be provided to use the values from Warnaar et al.
    skip_tests : list
        List of tests to skip.

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
        quality_outcomes_df, quality_values_df = (
            test_fraction_detected_breaths(
                native_peak_set,
                linked_timeseries,
                quality_outcomes_df,
                quality_values_df,
                n_peaks,
                cutoff,
                parameter_names,
            )
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
