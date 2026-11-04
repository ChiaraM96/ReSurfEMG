"""Tests for resurfemg.helper_functions.data_classes_quality_assessment.

These pin the behaviour of the TimeSeries quality-test helpers so the
refactor can't silently change which peaks get marked valid.

The numeric work lives in postprocessing.quality_assessment and has its own
tests. Here every qa.* call is replaced by a recording stub, so each test
checks one thing: what gets validated, what gets passed down, and where the
result ends up.

Gotcha: the module under test defines functions named ``test_*``. Never
``from ... import test_snr`` into this file or pytest will try to run them.
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from resurfemg.data_connector.data_classes import TimeSeries
from resurfemg.data_connector.peakset_class import PeaksSet
from resurfemg.helper_functions import data_classes_quality_assessment as dqa

FS = 100
N_SAMP = 2000
PEAK_IDXS = np.array([200, 600, 1000, 1400])
N_PEAKS = len(PEAK_IDXS)

TOLERANT_EMG = {
    "interpeak_distance": 1.1,
    "snr": 1.4,
    "aub": 40,
    "curve_fit": 30,
    "aub_window_s": 5 * FS,
    "bell_window_s": 5 * FS,
    "relative_aub_percentile": 75.0,
    "relative_aub_factor": 4.0,
    "relative_etp_upper_percentile": 95.0,
    "relative_etp_upper_factor": 10.0,
    "relative_etp_lower_percentile": 5.0,
    "relative_etp_lower_factor": 0.1,
}
STRICT_EMG = {
    **TOLERANT_EMG,
    "snr": 1.75,
    "aub": 30,
    "curve_fit": 25,
    "relative_aub_factor": 2.0,
}
POCC_DEFAULTS = {
    "consecutive_poccs": 0,
    "dP_up_10": 0.0,
    "dP_up_90": 2.0,
    "dP_up_90_norm": 0.8,
}
LINKED_COMPLETE = {
    "fraction_emg_breaths": 0.1,
    "delta_min": -0.5,
    "delta_max": 2.0,
}
EMG_TESTS = [
    "interpeak_distance",
    "snr",
    "aub",
    "curve_fit",
    "relative_aub",
    "relative_etp",
]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _peak_set(signal, t_data, idxs=PEAK_IDXS, full=True):
    peak_set = PeaksSet(signal, t_data, peak_idxs=idxs.copy())
    if full:
        peak_set.peak_df["start_idx"] = idxs - 50
        peak_set.peak_df["end_idx"] = idxs + 50
        peak_set.peak_df["time_product"] = np.array([1.0, 1.1, 0.9, 1.0])
        peak_set.peak_df["AUB"] = np.array([0.1, 0.2, 0.1, 0.15])
    return peak_set


@pytest.fixture
def emg_ts():
    """EMG TimeSeries with baseline, ECG peaks and a fully annotated set."""
    y = np.random.default_rng(0).normal(size=N_SAMP)
    ts = TimeSeries(y, fs=FS)
    ts["baseline"] = np.full(N_SAMP, 0.1)
    ts.peaks["breaths"] = _peak_set(y, ts.t_data)
    ts.peaks["ecg"] = PeaksSet(
        y, ts.t_data, peak_idxs=np.arange(50, N_SAMP, 80)
    )
    return ts


@pytest.fixture
def bare_ts():
    """EMG TimeSeries with only peak_idx, no baseline, no ECG."""
    y = np.zeros(N_SAMP)
    ts = TimeSeries(y, fs=FS)
    ts.peaks["breaths"] = _peak_set(y, ts.t_data, full=False)
    return ts


@pytest.fixture
def vent_ts():
    """Ventilator pressure TimeSeries with Poccs and ventilator breaths.

    Different fs from the EMG on purpose, to catch fs mix-ups.
    """
    fs_vent = 50
    y = np.zeros(N_SAMP // 2)
    ts = TimeSeries(y, fs=fs_vent)
    pocc_idxs = PEAK_IDXS // 2
    ts.peaks["Pocc"] = _peak_set(y, ts.t_data, idxs=pocc_idxs)
    ts.peaks["ventilator_breaths"] = PeaksSet(
        y, ts.t_data, peak_idxs=np.arange(25, N_SAMP // 2, 100)
    )
    ts.param["rr"] = 15.0
    return ts


class Recorder:
    """Stand-in for the qa module functions that remembers each call."""

    def __init__(self):
        self.calls = {}

    def stub(self, name, result):
        def _fn(*args, **kwargs):
            self.calls.setdefault(name, []).append((args, kwargs))
            return result(*args, **kwargs) if callable(result) else result

        return _fn

    def kwargs(self, name):
        assert name in self.calls, f"qa.{name} was not called"
        return self.calls[name][-1][1]

    def args(self, name):
        return self.calls[name][-1][0]


@pytest.fixture
def qa_stub(monkeypatch):
    """Replace every qa function the module uses with a recording stub.

    Outputs are chosen so each test has a distinct fingerprint, and peak 1
    fails SNR so validity aggregation can be checked.
    """
    rec = Recorder()
    n = N_PEAKS
    stubs = {
        "interpeak_dist": True,
        "snr_pseudo": np.array([2.0, 1.0, 3.0, 4.0]),
        "percentage_under_baseline": (
            np.ones(n, dtype=bool),
            np.full(n, 5.0),
            np.full(n, 0.2),
        ),
        "evaluate_bell_curve_error": (
            np.ones(n, dtype=bool),
            None,
            np.full(n, 7.0),
            np.full(n, -1.0),
            np.arange(3 * n, dtype=float).reshape(n, 3),
        ),
        "detect_local_high_aub": np.ones(n, dtype=bool),
        "detect_extreme_time_products": np.ones(n, dtype=bool),
        "detect_non_consecutive_manoeuvres": np.ones(n, dtype=bool),
        "pocc_quality": (
            np.ones(n, dtype=bool),
            np.vstack([np.full(n, 0.1), np.full(n, 0.2), np.full(n, 0.3)]),
        ),
        "evaluate_respiratory_rates": (0.8, None),
        "evaluate_event_timing": (np.ones(n, dtype=bool), np.full(n, 0.25)),
    }
    for name, result in stubs.items():
        monkeypatch.setattr(dqa.qa, name, rec.stub(name, result))
    return rec


def _emg_init(ts, cutoff=None, skip_tests=None, parameter_names=None):
    return dqa.initialize_emg_tests(
        ts, "breaths", cutoff, skip_tests, parameter_names
    )


def _dfs():
    df = pd.DataFrame({"peak_idx": PEAK_IDXS})
    return df.copy(), df.copy()


# --------------------------------------------------------------------------
# Import hygiene
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "first_import",
    [
        "resurfemg",
        "resurfemg.data_connector.data_classes",
        "resurfemg.helper_functions.data_classes_quality_assessment",
    ],
)
def test_module_imports_in_any_order(first_import):
    """data_classes imports this module, so it must not import data_classes
    back at runtime. Fresh interpreter, because sys.modules hides cycles."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {first_import}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


# --------------------------------------------------------------------------
# initialize_emg_tests
# --------------------------------------------------------------------------
class TestInitializeEmgTests:
    def test_unknown_peak_set_raises(self, emg_ts):
        with pytest.raises(KeyError, match="Non-existent PeaksSet key"):
            dqa.initialize_emg_tests(emg_ts, "nope", None, None, None)

    @pytest.mark.parametrize(
        "cutoff, expected",
        [
            (None, TOLERANT_EMG),
            ("tolerant", TOLERANT_EMG),
            ("strict", STRICT_EMG),
        ],
    )
    def test_preset_values(self, emg_ts, cutoff, expected):
        _, out_cutoff, *_ = _emg_init(emg_ts, cutoff)
        assert out_cutoff == expected

    def test_windows_scale_with_fs(self, emg_ts):
        emg_ts.param["fs"] = 2048
        _, out_cutoff, *_ = _emg_init(emg_ts, "tolerant")
        assert out_cutoff["aub_window_s"] == 5 * 2048
        assert out_cutoff["bell_window_s"] == 5 * 2048

    def test_preset_is_a_fresh_dict(self, emg_ts):
        """Mutating a returned preset must not leak into the next call."""
        _, first, *_ = _emg_init(emg_ts, "tolerant")
        first["snr"] = -999
        _, second, *_ = _emg_init(emg_ts, "tolerant")
        assert second["snr"] == 1.4

    def test_complete_dict_passes_through(self, emg_ts):
        cutoff = {k: float(v) for k, v in TOLERANT_EMG.items()}
        _, out_cutoff, *_ = _emg_init(emg_ts, dict(cutoff))
        assert out_cutoff == cutoff

    def test_dict_with_integer_thresholds_is_accepted(self, emg_ts):
        """The presets themselves use ints (aub=40, curve_fit=30), so a user
        copying them into a dict must not be rejected."""
        _, out_cutoff, *_ = _emg_init(emg_ts, dict(TOLERANT_EMG))
        assert out_cutoff["aub"] == 40

    def test_dict_with_numpy_float_is_accepted(self, emg_ts):
        cutoff = dict(TOLERANT_EMG, snr=np.float32(1.4))
        _emg_init(emg_ts, cutoff)

    @pytest.mark.parametrize(
        "missing", ["interpeak_distance", "snr", "aub", "curve_fit"]
    )
    def test_dict_missing_threshold_raises(self, emg_ts, missing):
        cutoff = dict(TOLERANT_EMG)
        del cutoff[missing]
        with pytest.raises(KeyError, match=missing):
            _emg_init(emg_ts, cutoff)

    @pytest.mark.parametrize(
        "missing", ["interpeak_distance", "snr", "aub", "curve_fit"]
    )
    def test_dict_missing_threshold_ok_when_skipped(self, emg_ts, missing):
        cutoff = {k: float(v) for k, v in TOLERANT_EMG.items()}
        del cutoff[missing]
        _emg_init(emg_ts, cutoff, skip_tests=[missing])

    @pytest.mark.parametrize(
        "missing, owner",
        [
            ("aub_window_s", "aub"),
            ("relative_aub_percentile", "relative_aub"),
            ("relative_aub_factor", "relative_aub"),
            ("relative_etp_upper_percentile", "relative_etp"),
            ("relative_etp_upper_factor", "relative_etp"),
            ("relative_etp_lower_percentile", "relative_etp"),
            ("relative_etp_lower_factor", "relative_etp"),
        ],
    )
    def test_dict_missing_aux_param(self, emg_ts, missing, owner):
        cutoff = {k: float(v) for k, v in TOLERANT_EMG.items()}
        del cutoff[missing]
        with pytest.raises(KeyError):
            _emg_init(emg_ts, dict(cutoff))
        _emg_init(emg_ts, dict(cutoff), skip_tests=[owner])

    @pytest.mark.parametrize("bad", ["lenient", "Tolerant", 1.4, ["strict"]])
    def test_invalid_cutoff_raises(self, emg_ts, bad):
        """Unknown presets used to fall through and blow up later with a
        confusing TypeError. They should be rejected up front."""
        with pytest.raises(ValueError):
            _emg_init(emg_ts, bad)

    def test_parameter_name_defaults(self, emg_ts):
        _, _, _, names, *_ = _emg_init(emg_ts)
        assert names == {
            "ecg": "ecg",
            "time_product": "time_product",
            "baseline": "baseline",
        }

    def test_custom_parameter_names_kept(self, emg_ts):
        _, _, _, names, *_ = _emg_init(
            emg_ts, parameter_names={"time_product": "ETPdi"}
        )
        assert names["time_product"] == "ETPdi"
        assert names["ecg"] == "ecg"

    def test_skip_tests_none_becomes_list(self, emg_ts):
        skip, *_ = _emg_init(emg_ts)
        assert skip == []

    def test_returns_peak_set_and_empty_frames(self, emg_ts):
        (_, _, peak_set, _, n_peaks, values_df, outcomes_df) = _emg_init(
            emg_ts
        )
        assert peak_set is emg_ts.peaks["breaths"]
        assert n_peaks == N_PEAKS
        for df in (values_df, outcomes_df):
            assert list(df.columns) == ["peak_idx"]
            np.testing.assert_array_equal(df["peak_idx"], PEAK_IDXS)

    def test_values_and_outcomes_frames_are_independent(self, emg_ts):
        *_, values_df, outcomes_df = _emg_init(emg_ts)
        values_df["snr"] = 1.0
        assert "snr" not in outcomes_df.columns


# --------------------------------------------------------------------------
# Individual EMG tests
# --------------------------------------------------------------------------
def _ctx(ts):
    (_, cutoff, peak_set, names, n_peaks, values_df, outcomes_df) = _emg_init(
        ts
    )
    return cutoff, peak_set, names, n_peaks, values_df, outcomes_df


class TestInterpeakDistance:
    def test_requires_ecg_peaks(self, bare_ts, qa_stub):
        cutoff, ps, names, n, _, out = _ctx(bare_ts)
        with pytest.raises(ValueError, match="ECG peaks"):
            dqa.test_interpeak_distance(bare_ts, ps, out, n, cutoff, names)

    def test_custom_ecg_name(self, emg_ts, qa_stub):
        emg_ts.peaks["ECG_R"] = emg_ts.peaks.pop("ecg")
        cutoff, ps, names, n, _, out = _ctx(emg_ts)
        names["ecg"] = "ECG_R"
        dqa.test_interpeak_distance(emg_ts, ps, out, n, cutoff, names)
        np.testing.assert_array_equal(
            qa_stub.kwargs("interpeak_dist")["ecg_peak_idxs"],
            emg_ts.peaks["ECG_R"].peak_df["peak_idx"],
        )

    def test_broadcasts_scalar_outcome(self, emg_ts, qa_stub):
        cutoff, ps, names, n, _, out = _ctx(emg_ts)
        out = dqa.test_interpeak_distance(emg_ts, ps, out, n, cutoff, names)
        kw = qa_stub.kwargs("interpeak_dist")
        assert kw["threshold"] == 1.1
        np.testing.assert_array_equal(kw["emg_peak_idxs"], PEAK_IDXS)
        np.testing.assert_array_equal(
            out["interpeak_distance"], [True] * N_PEAKS
        )


class TestSnr:
    def test_requires_baseline(self, bare_ts, qa_stub):
        cutoff, ps, names, _, vals, out = _ctx(bare_ts)
        with pytest.raises(ValueError, match="Baseline"):
            dqa.test_snr(bare_ts, ps, out, vals, cutoff, names)

    def test_writes_values_and_outcomes(self, emg_ts, qa_stub):
        cutoff, ps, names, _, vals, out = _ctx(emg_ts)
        out = dqa.test_snr(emg_ts, ps, out, vals, cutoff, names)
        kw = qa_stub.kwargs("snr_pseudo")
        assert kw["fs"] == FS
        np.testing.assert_array_equal(kw["baseline"], emg_ts["baseline"])
        np.testing.assert_array_equal(vals["snr"], [2.0, 1.0, 3.0, 4.0])
        np.testing.assert_array_equal(out["snr"], [True, False, True, True])

    def test_threshold_is_strict(self, emg_ts, qa_stub, monkeypatch):
        monkeypatch.setattr(
            dqa.qa, "snr_pseudo", lambda **_: np.array([1.4, 1.41, 1.39, 1.4])
        )
        cutoff, ps, names, _, vals, out = _ctx(emg_ts)
        out = dqa.test_snr(emg_ts, ps, out, vals, cutoff, names)
        np.testing.assert_array_equal(out["snr"], [False, True, False, False])


class TestAub:
    def test_requires_baseline(self, bare_ts, qa_stub):
        bare_ts.peaks["breaths"] = _peak_set(bare_ts["raw"], bare_ts.t_data)
        cutoff, ps, names, _, vals, out = _ctx(bare_ts)
        with pytest.raises(ValueError, match="Baseline"):
            dqa.test_aub(bare_ts, ps, out, vals, cutoff, names)

    @pytest.mark.parametrize("col", ["start_idx", "end_idx"])
    def test_requires_on_offsets(self, emg_ts, qa_stub, col):
        emg_ts.peaks["breaths"].peak_df.drop(columns=col, inplace=True)
        cutoff, ps, names, _, vals, out = _ctx(emg_ts)
        with pytest.raises(ValueError, match=col.split("_")[0]):
            dqa.test_aub(emg_ts, ps, out, vals, cutoff, names)

    def test_wiring(self, emg_ts, qa_stub):
        cutoff, ps, names, _, vals, out = _ctx(emg_ts)
        out, vals = dqa.test_aub(emg_ts, ps, out, vals, cutoff, names)
        kw = qa_stub.kwargs("percentage_under_baseline")
        assert kw["aub_threshold"] == 40
        assert kw["fs"] == FS
        np.testing.assert_array_equal(kw["start_idxs"], PEAK_IDXS - 50)
        np.testing.assert_array_equal(kw["end_idxs"], PEAK_IDXS + 50)
        np.testing.assert_array_equal(vals["aub"], [5.0] * N_PEAKS)
        np.testing.assert_array_equal(vals["aub_y_refs"], [0.2] * N_PEAKS)
        np.testing.assert_array_equal(out["aub"], [True] * N_PEAKS)


class TestCurveFits:
    def test_requires_baseline(self, bare_ts, qa_stub):
        bare_ts.peaks["breaths"] = _peak_set(bare_ts["raw"], bare_ts.t_data)
        cutoff, ps, names, _, vals, out = _ctx(bare_ts)
        with pytest.raises(ValueError, match="Baseline"):
            dqa.test_curve_fits(bare_ts, ps, out, vals, cutoff, names)

    def test_requires_baseline_not_none(self, emg_ts, qa_stub):
        emg_ts["baseline"] = None
        cutoff, ps, names, _, vals, out = _ctx(emg_ts)
        with pytest.raises(ValueError, match="Baseline"):
            dqa.test_curve_fits(emg_ts, ps, out, vals, cutoff, names)

    @pytest.mark.parametrize("col", ["start_idx", "end_idx", "time_product"])
    def test_requires_columns(self, emg_ts, qa_stub, col):
        emg_ts.peaks["breaths"].peak_df.drop(columns=col, inplace=True)
        cutoff, ps, names, _, vals, out = _ctx(emg_ts)
        with pytest.raises(ValueError):
            dqa.test_curve_fits(emg_ts, ps, out, vals, cutoff, names)

    def test_wiring_and_peak_df_side_effects(self, emg_ts, qa_stub):
        cutoff, ps, names, _, vals, out = _ctx(emg_ts)
        out, vals, ps_out = dqa.test_curve_fits(
            emg_ts, ps, out, vals, cutoff, names
        )
        kw = qa_stub.kwargs("evaluate_bell_curve_error")
        assert kw["bell_window_s"] == 5 * FS
        assert kw["bell_threshold"] == 30
        assert ps_out is ps
        np.testing.assert_array_equal(ps.peak_df["bell_y_min"], [-1.0] * 4)
        params = np.arange(12, dtype=float).reshape(4, 3)
        for i, p in enumerate("abc"):
            np.testing.assert_array_equal(
                ps.peak_df[f"bell_{p}"], params[:, i]
            )
        np.testing.assert_array_equal(vals["bell"], [7.0] * N_PEAKS)
        np.testing.assert_array_equal(out["bell"], [True] * N_PEAKS)


class TestRelativeAub:
    def test_requires_aub_column(self, emg_ts, qa_stub):
        emg_ts.peaks["breaths"].peak_df.drop(columns="AUB", inplace=True)
        cutoff, ps, _, _, _, out = _ctx(emg_ts)
        with pytest.raises(ValueError, match="AUB"):
            dqa.test_relative_aub(ps, out, cutoff)

    def test_wiring(self, emg_ts, qa_stub):
        cutoff, ps, _, _, _, out = _ctx(emg_ts)
        out = dqa.test_relative_aub(ps, out, cutoff)
        kw = qa_stub.kwargs("detect_local_high_aub")
        assert kw["threshold_percentile"] == 75.0
        assert kw["threshold_factor"] == 4.0
        np.testing.assert_array_equal(kw["aubs"], ps.peak_df["AUB"])
        assert "relative_aub" in out.columns


class TestRelativeEtp:
    def test_requires_time_products(self, emg_ts, qa_stub):
        cutoff, ps, names, _, _, out = _ctx(emg_ts)
        names["time_product"] = "ETPdi"
        with pytest.raises(ValueError, match="ETP"):
            dqa.test_relative_etp(ps, out, cutoff, names)

    def test_wiring(self, emg_ts, qa_stub):
        cutoff, ps, names, _, _, out = _ctx(emg_ts)
        out = dqa.test_relative_etp(ps, out, cutoff, names)
        kw = qa_stub.kwargs("detect_extreme_time_products")
        np.testing.assert_array_equal(
            kw.pop("time_products"), ps.peak_df["time_product"]
        )
        assert kw == {
            "upper_percentile": 95.0,
            "upper_factor": 10.0,
            "lower_percentile": 5.0,
            "lower_factor": 0.1,
        }
        assert "relative_etp" in out.columns


# --------------------------------------------------------------------------
# Pocc
# --------------------------------------------------------------------------
class TestInitializePoccTests:
    def test_unknown_peak_set_raises(self, vent_ts):
        with pytest.raises(KeyError):
            dqa.initialize_pocc_tests(vent_ts, "nope", None, None, None)

    @pytest.mark.parametrize("cutoff", [None, "tolerant", "strict"])
    def test_presets(self, vent_ts, cutoff):
        _, out, *_ = dqa.initialize_pocc_tests(
            vent_ts, "Pocc", cutoff, None, None
        )
        assert out == POCC_DEFAULTS

    @pytest.mark.parametrize(
        "missing, owner",
        [
            ("consecutive_poccs", "consecutive_poccs"),
            ("dP_up_10", "pocc_upslope"),
            ("dP_up_90", "pocc_upslope"),
            ("dP_up_90_norm", "pocc_upslope"),
        ],
    )
    def test_dict_missing_key(self, vent_ts, missing, owner):
        cutoff = dict(POCC_DEFAULTS)
        del cutoff[missing]
        with pytest.raises(KeyError, match=owner):
            dqa.initialize_pocc_tests(
                vent_ts, "Pocc", dict(cutoff), None, None
            )
        dqa.initialize_pocc_tests(vent_ts, "Pocc", dict(cutoff), [owner], None)

    def test_parameter_name_defaults(self, vent_ts):
        _, _, _, names, n, vals, out = dqa.initialize_pocc_tests(
            vent_ts, "Pocc", None, None, {"time_product": "PTP"}
        )
        assert names == {
            "ventilator_breaths": "ventilator_breaths",
            "time_product": "PTP",
            "AUB": "AUB",
        }
        assert n == N_PEAKS
        assert vals is not out


class TestPoccTests:
    def _ctx(self, vent_ts):
        (_, cutoff, ps, names, _, vals, out) = dqa.initialize_pocc_tests(
            vent_ts, "Pocc", None, None, None
        )
        return cutoff, ps, names, vals, out

    def test_consecutive_requires_vent_breaths(self, vent_ts, qa_stub):
        del vent_ts.peaks["ventilator_breaths"]
        _, ps, names, _, out = self._ctx(vent_ts)
        with pytest.raises(ValueError, match="Ventilator breaths"):
            dqa.test_consecutive_poccs(vent_ts, ps, out, names)

    def test_consecutive_wiring(self, vent_ts, qa_stub):
        _, ps, names, _, out = self._ctx(vent_ts)
        out = dqa.test_consecutive_poccs(vent_ts, ps, out, names)
        kw = qa_stub.kwargs("detect_non_consecutive_manoeuvres")
        np.testing.assert_array_equal(
            kw["ventilator_breath_idxs"],
            vent_ts.peaks["ventilator_breaths"].peak_df["peak_idx"],
        )
        np.testing.assert_array_equal(kw["manoeuvres_idxs"], PEAK_IDXS // 2)
        assert "consecutive_poccs" in out.columns

    @pytest.mark.parametrize("col", ["end_idx", "time_product"])
    def test_upslope_requires_columns(self, vent_ts, qa_stub, col):
        vent_ts.peaks["Pocc"].peak_df.drop(columns=col, inplace=True)
        cutoff, ps, names, vals, out = self._ctx(vent_ts)
        with pytest.raises(ValueError):
            dqa.test_pocc_upslope(vent_ts, ps, out, vals, cutoff, names)

    def test_upslope_wiring(self, vent_ts, qa_stub):
        cutoff, ps, names, vals, out = self._ctx(vent_ts)
        out, vals = dqa.test_pocc_upslope(
            vent_ts, ps, out, vals, cutoff, names
        )
        kw = qa_stub.kwargs("pocc_quality")
        assert kw["p_vent_signal"] is vent_ts["raw"]
        assert (
            kw["dp_up_10_threshold"],
            kw["dp_up_90_threshold"],
            kw["dp_up_90_norm_threshold"],
        ) == (0.0, 2.0, 0.8)
        np.testing.assert_array_equal(vals["dP_up_10"], [0.1] * N_PEAKS)
        np.testing.assert_array_equal(vals["dP_up_90"], [0.2] * N_PEAKS)
        np.testing.assert_array_equal(vals["dP_up_90_norm"], [0.3] * N_PEAKS)
        assert "pocc_upslope" in out.columns


# --------------------------------------------------------------------------
# Linked peak sets
# --------------------------------------------------------------------------
def _linked_init(emg_ts, vent_ts, cutoff=None, skip=None, names=None):
    return dqa.initialize_linked_peaks_tests(
        emg_ts, "breaths", vent_ts, "Pocc", cutoff, skip, names
    )


class TestInitializeLinkedPeaksTests:
    def test_unknown_native_raises(self, emg_ts, vent_ts):
        with pytest.raises(KeyError, match="Non-existent PeaksSet"):
            dqa.initialize_linked_peaks_tests(
                emg_ts, "nope", vent_ts, "Pocc", None, None, None
            )

    def test_unknown_linked_raises(self, emg_ts, vent_ts):
        with pytest.raises(KeyError, match="linked"):
            dqa.initialize_linked_peaks_tests(
                emg_ts, "breaths", vent_ts, "nope", None, None, None
            )

    @pytest.mark.parametrize(
        "cutoff, fraction", [(None, 0.1), ("tolerant", 0.1), ("strict", 0.5)]
    )
    def test_presets(self, emg_ts, vent_ts, cutoff, fraction):
        _, out, *_ = _linked_init(emg_ts, vent_ts, cutoff)
        assert out == {
            "fraction_emg_breaths": fraction,
            "delta_min": -0.5,
            "delta_max": 2.0,
        }

    def test_complete_dict_accepted(self, emg_ts, vent_ts):
        _, out, *_ = _linked_init(emg_ts, vent_ts, dict(LINKED_COMPLETE))
        assert out == LINKED_COMPLETE

    @pytest.mark.parametrize("missing", list(LINKED_COMPLETE))
    def test_dict_missing_key_raises(self, emg_ts, vent_ts, missing):
        cutoff = dict(LINKED_COMPLETE)
        del cutoff[missing]
        with pytest.raises(KeyError):
            _linked_init(emg_ts, vent_ts, cutoff)

    @pytest.mark.parametrize(
        "missing, owner",
        [
            ("fraction_emg_breaths", "fraction_emg_breaths"),
            ("delta_min", "event_timing"),
            ("delta_max", "event_timing"),
        ],
    )
    def test_dict_missing_key_ok_when_skipped(
        self, emg_ts, vent_ts, missing, owner
    ):
        cutoff = dict(LINKED_COMPLETE)
        del cutoff[missing]
        _linked_init(emg_ts, vent_ts, cutoff, skip=[owner])

    @pytest.mark.parametrize("bad", ["lenient", 0.5])
    def test_invalid_cutoff_raises(self, emg_ts, vent_ts, bad):
        with pytest.raises(ValueError):
            _linked_init(emg_ts, vent_ts, bad)

    def test_returns(self, emg_ts, vent_ts):
        (skip, _, ps, linked, names, n, vals, out) = _linked_init(
            emg_ts, vent_ts
        )
        assert skip == []
        assert ps is emg_ts.peaks["breaths"]
        assert linked is vent_ts.peaks["Pocc"]
        assert names == {
            "ventilator_breaths": "ventilator_breaths",
            "rr": "rr",
        }
        assert n == N_PEAKS
        assert vals is not out


class TestLinkedTests:
    def test_fraction_wiring(self, emg_ts, vent_ts, qa_stub):
        (_, cutoff, ps, _, names, n, vals, out) = _linked_init(emg_ts, vent_ts)
        out, vals = dqa.test_fraction_detected_breaths(
            ps, vent_ts, out, vals, n, cutoff, names
        )
        kw = qa_stub.kwargs("evaluate_respiratory_rates")
        assert kw["rr_vent"] == 15.0
        assert kw["t_emg"] == max(emg_ts.t_data)
        np.testing.assert_array_equal(vals["detected_fraction"], [0.8] * 4)
        np.testing.assert_array_equal(out["detected_fraction"], [True] * 4)

    def test_fraction_custom_rr_name(self, emg_ts, vent_ts, qa_stub):
        vent_ts.param["rr_occ"] = 3.0
        (_, cutoff, ps, _, names, n, vals, out) = _linked_init(
            emg_ts, vent_ts, names={"rr": "rr_occ"}
        )
        dqa.test_fraction_detected_breaths(
            ps, vent_ts, out, vals, n, cutoff, names
        )
        assert qa_stub.kwargs("evaluate_respiratory_rates")["rr_vent"] == 3.0

    def test_fraction_threshold_is_strict(self, emg_ts, vent_ts, monkeypatch):
        monkeypatch.setattr(
            dqa.qa, "evaluate_respiratory_rates", lambda **_: (0.5, None)
        )
        (_, cutoff, ps, _, names, n, vals, out) = _linked_init(
            emg_ts, vent_ts, "strict"
        )
        out, _ = dqa.test_fraction_detected_breaths(
            ps, vent_ts, out, vals, n, cutoff, names
        )
        assert not out["detected_fraction"].any()

    def test_event_timing_uses_each_series_fs(self, emg_ts, vent_ts, qa_stub):
        (_, cutoff, ps, linked, _, _, vals, out) = _linked_init(
            emg_ts, vent_ts
        )
        out, vals = dqa.test_event_timing(
            emg_ts, ps, vent_ts, linked, out, vals, cutoff
        )
        args = qa_stub.args("evaluate_event_timing")
        # Same events in seconds: EMG at fs=100, vent at fs=50 with idx/2.
        np.testing.assert_allclose(args[0], PEAK_IDXS / FS)
        np.testing.assert_allclose(args[1], PEAK_IDXS / FS)
        kw = qa_stub.kwargs("evaluate_event_timing")
        assert (kw["delta_min"], kw["delta_max"]) == (-0.5, 2.0)
        np.testing.assert_array_equal(vals["event_delta_time"], [0.25] * 4)
        assert "event_timing" in out.columns


# --------------------------------------------------------------------------
# Orchestrators
# --------------------------------------------------------------------------
QA_FN_FOR_EMG_TEST = {
    "interpeak_distance": "interpeak_dist",
    "snr": "snr_pseudo",
    "aub": "percentage_under_baseline",
    "curve_fit": "evaluate_bell_curve_error",
    "relative_aub": "detect_local_high_aub",
    "relative_etp": "detect_extreme_time_products",
}


class TestEmgQuality:
    def test_runs_all_and_aggregates_validity(self, emg_ts, qa_stub):
        dqa.test_emg_quality(emg_ts, "breaths", verbose=False)
        ps = emg_ts.peaks["breaths"]
        for col in [
            "interpeak_distance",
            "snr",
            "aub",
            "bell",
            "relative_aub",
            "relative_etp",
        ]:
            assert col in ps.quality_outcomes_df.columns
        for col in ["snr", "aub", "aub_y_refs", "bell"]:
            assert col in ps.quality_values_df.columns
        # Only peak 1 fails (SNR 1.0 < 1.4).
        np.testing.assert_array_equal(
            ps.peak_df["valid"], [True, False, True, True]
        )

    def test_strict_preset_reaches_qa(self, emg_ts, qa_stub):
        dqa.test_emg_quality(emg_ts, "breaths", cutoff="strict", verbose=False)
        kw = qa_stub.kwargs("percentage_under_baseline")
        assert kw["aub_threshold"] == 30

    @pytest.mark.parametrize(
        "skipped", [t for t in EMG_TESTS if t != "interpeak_distance"]
    )
    def test_skip(self, emg_ts, qa_stub, skipped):
        dqa.test_emg_quality(
            emg_ts, "breaths", skip_tests=[skipped], verbose=False
        )
        assert QA_FN_FOR_EMG_TEST[skipped] not in qa_stub.calls
        for other in set(EMG_TESTS) - {skipped}:
            assert QA_FN_FOR_EMG_TEST[other] in qa_stub.calls

    def test_skip_interpeak_distance(self, emg_ts, qa_stub):
        dqa.test_emg_quality(
            emg_ts, "breaths", skip_tests=["interpeak_distance"], verbose=False
        )
        assert "interpeak_dist" not in qa_stub.calls

    def test_verbose(self, emg_ts, qa_stub, capsys):
        dqa.test_emg_quality(emg_ts, "breaths", verbose=True)
        assert "Test outcomes" in capsys.readouterr().out
        dqa.test_emg_quality(emg_ts, "breaths", verbose=False)
        assert capsys.readouterr().out == ""

    def test_rerun_overwrites_instead_of_duplicating(self, emg_ts, qa_stub):
        dqa.test_emg_quality(emg_ts, "breaths", verbose=False)
        dqa.test_emg_quality(emg_ts, "breaths", verbose=False)
        cols = emg_ts.peaks["breaths"].quality_outcomes_df.columns
        assert not cols.duplicated().any()

    def test_caller_parameter_names_not_lost(self, emg_ts, qa_stub):
        ps = emg_ts.peaks["breaths"]
        ps.peak_df["ETPdi"] = ps.peak_df.pop("time_product")
        dqa.test_emg_quality(
            emg_ts,
            "breaths",
            verbose=False,
            parameter_names={"time_product": "ETPdi"},
        )
        np.testing.assert_array_equal(
            qa_stub.kwargs("detect_extreme_time_products")["time_products"],
            ps.peak_df["ETPdi"],
        )


class TestPoccQuality:
    def test_runs_all(self, vent_ts, qa_stub):
        dqa.test_pocc_quality(vent_ts, "Pocc", verbose=False)
        ps = vent_ts.peaks["Pocc"]
        assert {"consecutive_poccs", "pocc_upslope"} <= set(
            ps.quality_outcomes_df.columns
        )
        assert {"dP_up_10", "dP_up_90", "dP_up_90_norm"} <= set(
            ps.quality_values_df.columns
        )
        assert ps.peak_df["valid"].all()

    @pytest.mark.parametrize(
        "skipped, fn",
        [
            ("consecutive_poccs", "detect_non_consecutive_manoeuvres"),
            ("pocc_upslope", "pocc_quality"),
        ],
    )
    def test_skip(self, vent_ts, qa_stub, skipped, fn):
        dqa.test_pocc_quality(
            vent_ts, "Pocc", skip_tests=[skipped], verbose=False
        )
        assert fn not in qa_stub.calls


class TestLinkedPeakSets:
    def test_runs_all(self, emg_ts, vent_ts, qa_stub):
        dqa.test_linked_peak_sets(
            emg_ts, "breaths", vent_ts, "Pocc", verbose=False
        )
        ps = emg_ts.peaks["breaths"]
        assert {"detected_fraction", "event_timing"} <= set(
            ps.quality_outcomes_df.columns
        )
        assert {"detected_fraction", "event_delta_time"} <= set(
            ps.quality_values_df.columns
        )

    def test_dict_cutoff(self, emg_ts, vent_ts, qa_stub):
        """Mirrors the call in data_classes_test.py."""
        dqa.test_linked_peak_sets(
            emg_ts,
            "breaths",
            vent_ts,
            "Pocc",
            verbose=False,
            cutoff={
                "fraction_emg_breaths": 0.1,
                "delta_min": 0.2,
                "delta_max": 0.6,
            },
        )
        assert qa_stub.kwargs("evaluate_event_timing")["delta_max"] == 0.6

    def test_strict(self, emg_ts, vent_ts, qa_stub):
        dqa.test_linked_peak_sets(
            emg_ts, "breaths", vent_ts, "Pocc", cutoff="strict", verbose=False
        )

    @pytest.mark.parametrize(
        "skipped, fn",
        [
            ("fraction_emg_breaths", "evaluate_respiratory_rates"),
            ("event_timing", "evaluate_event_timing"),
        ],
    )
    def test_skip(self, emg_ts, vent_ts, qa_stub, skipped, fn):
        dqa.test_linked_peak_sets(
            emg_ts,
            "breaths",
            vent_ts,
            "Pocc",
            skip_tests=[skipped],
            verbose=False,
        )
        assert fn not in qa_stub.calls
