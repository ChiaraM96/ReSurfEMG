"""Tests for the PeaksSet class of the resurfemg library."""

import numpy as np
import pandas as pd
import pytest

from resurfemg.data_connector.peakset_class import PeaksSet
from resurfemg.postprocessing.event_detection import (
    onoffpeak_baseline_crossing,
    onoffpeak_slope_extrapolation,
)

FS = 100.0  # float on purpose: the default slope window must still be an int
T = np.arange(0, 10, 1 / FS)
# Five positive half-waves around a zero baseline, peaking at 0.5, 2.5, ... s.
SIGNAL = np.sin(2 * np.pi * 0.5 * (T + 0.01))
PEAK_IDXS = np.array([49, 249, 449, 649, 849])
# Raised-cosine bumps that touch zero, for slope extrapolation.
ENVELOPE = np.clip(SIGNAL, 0, None) ** 2


@pytest.fixture
def peaks() -> PeaksSet:
    return PeaksSet(SIGNAL, T, PEAK_IDXS)


def _tests_df(peak_idxs, **columns) -> pd.DataFrame:
    return pd.DataFrame({"peak_idx": peak_idxs, **columns})


# --- __init__ ---------------------------------------------------------------


def test_init_stores_arrays(peaks):
    np.testing.assert_array_equal(peaks.signal, SIGNAL)
    np.testing.assert_array_equal(peaks.t_data, T)
    np.testing.assert_array_equal(peaks["peak_idx"], PEAK_IDXS)
    assert peaks.time_products is None


@pytest.mark.parametrize(
    "convert", [list, tuple, np.asarray], ids=["list", "tuple", "array"]
)
def test_init_accepts_array_likes(convert):
    p = PeaksSet(convert(SIGNAL), convert(T), convert(PEAK_IDXS))
    assert isinstance(p.signal, np.ndarray)
    assert isinstance(p.t_data, np.ndarray)
    np.testing.assert_array_equal(p["peak_idx"], PEAK_IDXS)


def test_init_without_peaks_is_empty():
    p = PeaksSet(SIGNAL, T)
    assert len(p) == 0
    for df in (p.peak_df, p.quality_values_df, p.quality_outcomes_df):
        assert list(df.columns) == ["peak_idx"]
        assert df.empty


def test_init_dataframes_are_independent(peaks):
    peaks.quality_values_df["x"] = 1
    assert "x" not in peaks.quality_outcomes_df
    assert "x" not in peaks.peak_df


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signal": 5},
        {"signal": "abc"},
        {"t_data": None},
        {"peak_idxs": 3.0},
        {"peak_idxs": "abc"},
    ],
    ids=["signal_int", "signal_str", "t_data_none", "peaks_float", "peaks_str"],
)
def test_init_rejects_non_array_likes(kwargs):
    with pytest.raises(TypeError):
        PeaksSet(**{"signal": SIGNAL, "t_data": T, "peak_idxs": PEAK_IDXS, **kwargs})


def test_init_rejects_2d_peak_idxs():
    with pytest.raises(ValueError):
        PeaksSet(SIGNAL, T, np.array([[1, 2], [3, 4]]))


# --- dunder methods ---------------------------------------------------------


def test_len(peaks):
    assert len(peaks) == len(PEAK_IDXS)


def test_getitem_returns_numpy(peaks):
    assert isinstance(peaks["peak_idx"], np.ndarray)


def test_getitem_missing_key_raises(peaks):
    with pytest.raises(KeyError):
        peaks["nope"]


def test_str_matches_peak_df(peaks):
    assert str(peaks) == str(peaks.peak_df)


def test_keys(peaks):
    assert list(peaks.keys()) == ["peak_idx"]


# --- detect_on_offset -------------------------------------------------------


@pytest.mark.parametrize("method", ["default", "baseline_crossing"])
def test_detect_baseline_crossing_matches_event_detection(peaks, method):
    peaks.detect_on_offset(method=method)
    start, end, _, _, valid = onoffpeak_baseline_crossing(
        SIGNAL, np.zeros(SIGNAL.shape), PEAK_IDXS
    )
    np.testing.assert_array_equal(peaks["start_idx"], start)
    np.testing.assert_array_equal(peaks["end_idx"], end)
    np.testing.assert_array_equal(peaks["valid"], valid)


def test_detect_brackets_each_peak(peaks):
    peaks.detect_on_offset()
    assert np.all(peaks["start_idx"] <= peaks["peak_idx"])
    assert np.all(peaks["peak_idx"] <= peaks["end_idx"])


def test_detect_uses_custom_baseline(peaks):
    baseline = np.full(SIGNAL.shape, 0.5)
    peaks.detect_on_offset(baseline=baseline)
    start, end, *_ = onoffpeak_baseline_crossing(SIGNAL, baseline, PEAK_IDXS)
    np.testing.assert_array_equal(peaks["start_idx"], start)
    np.testing.assert_array_equal(peaks["end_idx"], end)


def test_detect_records_baseline_detection_outcome(peaks):
    peaks.detect_on_offset()
    assert "baseline_detection" in peaks.quality_outcomes_df
    np.testing.assert_array_equal(
        peaks.quality_outcomes_df["baseline_detection"], peaks["valid"]
    )


def test_detect_slope_extrapolation_matches_event_detection():
    p = PeaksSet(ENVELOPE, T, PEAK_IDXS)
    p.detect_on_offset(method="slope_extrapolation", fs=FS, slope_window_s=10)
    start, end, _, _, valid = onoffpeak_slope_extrapolation(ENVELOPE, FS, PEAK_IDXS, 10)
    np.testing.assert_array_equal(p["start_idx"], start)
    np.testing.assert_array_equal(p["end_idx"], end)
    np.testing.assert_array_equal(p["valid"], valid)


def test_detect_slope_extrapolation_default_window_is_fs_over_5():
    p = PeaksSet(ENVELOPE, T, PEAK_IDXS)
    p.detect_on_offset(method="slope_extrapolation", fs=FS)
    start, end, *_ = onoffpeak_slope_extrapolation(
        ENVELOPE, FS, PEAK_IDXS, int(FS // 5)
    )
    np.testing.assert_array_equal(p["start_idx"], start)
    np.testing.assert_array_equal(p["end_idx"], end)


def test_detect_slope_extrapolation_requires_fs(peaks):
    with pytest.raises(ValueError):
        peaks.detect_on_offset(method="slope_extrapolation")


def test_detect_unknown_method_raises(peaks):
    with pytest.raises(KeyError):
        peaks.detect_on_offset(method="nope")


def test_detect_keeps_earlier_failed_tests(peaks):
    peaks.evaluate_validity(_tests_df(PEAK_IDXS, snr=[False, True, True, True, True]))
    peaks.detect_on_offset()
    assert not peaks["valid"][0]


# --- update_test_outcomes ---------------------------------------------------


def test_update_test_outcomes_stores_values(peaks):
    peaks.update_test_outcomes(_tests_df(PEAK_IDXS, snr=[1.0, 2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(
        peaks.quality_values_df["snr"], [1.0, 2.0, 3.0, 4.0, 5.0]
    )


def test_update_test_outcomes_keeps_earlier_columns(peaks):
    peaks.update_test_outcomes(_tests_df(PEAK_IDXS, snr=np.ones(5)))
    peaks.update_test_outcomes(_tests_df(PEAK_IDXS, aub=np.zeros(5)))
    assert {"peak_idx", "snr", "aub"} <= set(peaks.quality_values_df.columns)


def test_update_test_outcomes_overwrites_existing_column(peaks):
    peaks.update_test_outcomes(_tests_df(PEAK_IDXS, snr=np.ones(5)))
    peaks.update_test_outcomes(_tests_df(PEAK_IDXS, snr=np.full(5, 2.0)))
    np.testing.assert_array_equal(peaks.quality_values_df["snr"], np.full(5, 2.0))
    assert list(peaks.quality_values_df.columns).count("snr") == 1


def test_update_test_outcomes_leaves_validity_alone(peaks):
    peaks.update_test_outcomes(_tests_df(PEAK_IDXS, snr=np.zeros(5)))
    assert "valid" not in peaks.keys()
    assert list(peaks.quality_outcomes_df.columns) == ["peak_idx"]


@pytest.mark.parametrize(
    "bad_idxs", [PEAK_IDXS + 1, PEAK_IDXS[:-1]], ids=["shifted", "shorter"]
)
def test_update_test_outcomes_mismatched_peaks_raise(peaks, bad_idxs):
    with pytest.raises(ValueError):
        peaks.update_test_outcomes(_tests_df(bad_idxs, snr=np.ones(len(bad_idxs))))


# --- evaluate_validity ------------------------------------------------------


def test_evaluate_validity_all_tests_must_pass(peaks):
    peaks.evaluate_validity(
        _tests_df(
            PEAK_IDXS,
            snr=[True, False, True, True, True],
            aub=[True, True, False, True, True],
        )
    )
    np.testing.assert_array_equal(peaks["valid"], [True, False, False, True, True])


def test_evaluate_validity_keeps_earlier_columns(peaks):
    peaks.evaluate_validity(_tests_df(PEAK_IDXS, snr=np.ones(5, dtype=bool)))
    peaks.evaluate_validity(_tests_df(PEAK_IDXS, aub=np.ones(5, dtype=bool)))
    assert {"peak_idx", "snr", "aub"} <= set(peaks.quality_outcomes_df.columns)


def test_evaluate_validity_keeps_earlier_failures(peaks):
    peaks.evaluate_validity(_tests_df(PEAK_IDXS, snr=[False, True, True, True, True]))
    peaks.evaluate_validity(_tests_df(PEAK_IDXS, aub=np.ones(5, dtype=bool)))
    assert not peaks["valid"][0]


def test_evaluate_validity_overwrites_existing_column(peaks):
    peaks.evaluate_validity(_tests_df(PEAK_IDXS, snr=np.zeros(5, dtype=bool)))
    peaks.evaluate_validity(_tests_df(PEAK_IDXS, snr=np.ones(5, dtype=bool)))
    assert peaks["valid"].all()
    assert list(peaks.quality_outcomes_df.columns).count("snr") == 1


@pytest.mark.parametrize(
    "bad_idxs", [PEAK_IDXS + 1, PEAK_IDXS[:-1]], ids=["shifted", "shorter"]
)
def test_evaluate_validity_mismatched_peaks_raise(peaks, bad_idxs):
    with pytest.raises(ValueError):
        peaks.evaluate_validity(
            _tests_df(bad_idxs, snr=np.ones(len(bad_idxs), dtype=bool))
        )


# --- sanitize ---------------------------------------------------------------


def test_sanitize_drops_invalid_peaks_everywhere(peaks):
    peaks.update_test_outcomes(_tests_df(PEAK_IDXS, snr=[1.0, 2.0, 3.0, 4.0, 5.0]))
    peaks.evaluate_validity(_tests_df(PEAK_IDXS, snr=[True, False, True, False, True]))
    peaks.sanitize()

    kept = PEAK_IDXS[[0, 2, 4]]
    for df in (peaks.peak_df, peaks.quality_values_df, peaks.quality_outcomes_df):
        np.testing.assert_array_equal(df["peak_idx"], kept)
        assert list(df.index) == [0, 1, 2]
    np.testing.assert_array_equal(peaks.quality_values_df["snr"], [1.0, 3.0, 5.0])


def test_sanitize_all_valid_is_noop(peaks):
    peaks.evaluate_validity(_tests_df(PEAK_IDXS, snr=np.ones(5, dtype=bool)))
    before = peaks.peak_df.copy()
    peaks.sanitize()
    pd.testing.assert_frame_equal(peaks.peak_df, before)


def test_sanitized_set_accepts_new_tests(peaks):
    peaks.evaluate_validity(_tests_df(PEAK_IDXS, snr=[True, False, True, True, True]))
    peaks.sanitize()
    kept = peaks["peak_idx"]
    peaks.evaluate_validity(_tests_df(kept, aub=np.ones(len(kept), dtype=bool)))
    assert peaks["valid"].all()
