"""sanity tests for the preprocessing functions, including filtering,
ecg removal and envelope calculation"""

import multiprocessing
import unittest
import os
from math import pi
import numpy as np
import pytest
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.preprocessing.filtering import emg_bandpass_butter
import resurfemg.helper_functions.math_operations as mo

sample_emg = os.path.join(
    os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
    "test_data",
    "emg_data_synth_quiet_breathing.Poly5",
)


class TestArrayMath(unittest.TestCase):
    def test_scale_arrays(self):
        sample_read = Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(
            sample_read.samples, 1, 500, sample_read.sample_rate
        )
        new_emg = mo.scale_arrays(sample_emg_filtered, 3, 0)
        self.assertEqual(
            (new_emg.shape),
            (sample_emg_filtered.shape),
        )

    def test_zero_one_for_jumps_base(self):
        sample_read = Poly5Reader(sample_emg)
        sample_emg_filtered = emg_bandpass_butter(
            sample_read.samples, 1, 500, sample_read.sample_rate
        )
        new_emg = mo.zero_one_for_jumps_base(
            sample_emg_filtered[0], sample_emg_filtered[0].mean()
        )
        new_emg = np.array(np.vstack((new_emg, new_emg)))
        self.assertEqual(
            (new_emg.shape[1]),
            (sample_emg_filtered.shape[1]),
        )


class TestDerivative(unittest.TestCase):
    def test_derivative(self):
        fs = 100
        t = np.array([i / fs for i in range(fs * 1)])
        y_t = np.sin(t * 2 * pi)
        dy_dt_ref = 2 * pi * np.cos(t * 2 * pi)[:-1]
        dy_dt_fun = mo.derivative(y_t, fs)
        error = np.sum(np.abs((dy_dt_ref - dy_dt_fun))) / (
            np.max(np.abs(dy_dt_ref)) * len(t) - 1
        )

        self.assertLess(error, 0.05)


# ---------------------------------------------------------------------------
# Characterisation tests for the math_operations refactor.
# Expected values are written out by hand and were checked against the
# pre-refactor implementation, so any change in behaviour shows up here.
# ---------------------------------------------------------------------------
R = mo.Range


class TestRange:
    def test_is_a_plain_tuple(self):
        r = R(2, 7)
        assert (r.start, r.end) == (2, 7)
        assert r == (2, 7)
        assert r._replace(end=9) == R(2, 9)
        assert r._asdict() == {"start": 2, "end": 7}
        assert repr(r) == "Range(start=2, end=7)"

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ((0, 5), (3, 9), True),  # partial overlap
            ((3, 9), (0, 5), True),
            ((0, 9), (3, 5), True),  # contains
            ((3, 5), (0, 9), True),  # contained
            ((0, 5), (0, 5), True),  # identical
            ((0, 2), (4, 6), False),  # disjoint
            ((4, 6), (0, 2), False),
            # Touching ends count as overlapping, in both directions.
            ((0, 5), (5, 9), True),
            ((5, 9), (0, 5), False),
        ],
    )
    def test_intersects(self, a, b, expected):
        assert R(*a).intersects(R(*b)) is expected

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ((0, 2), (3, 5), True),
            ((0, 3), (3, 5), False),  # strict
            ((3, 5), (0, 2), False),
        ],
    )
    def test_precedes(self, a, b, expected):
        assert R(*a).precedes(R(*b)) is expected

    def test_to_slice_casts_floats(self):
        """ranges_of builds Ranges from np.float64, so this must cope."""
        s = R(np.float64(2.0), np.float64(5.0)).to_slice()
        assert s == slice(2, 5)
        assert isinstance(s.start, int) and isinstance(s.stop, int)


class TestZeroOneForJumpsBase:
    def test_values(self):
        out = mo.zero_one_for_jumps_base(np.array([0.1, 0.5, 0.9, 0.4]), 0.5)
        assert out == [0, 1, 1, 0]  # at or above cut-off -> 1

    def test_returns_ints_not_bools(self):
        """True == 1 in Python, so the value test alone can't catch a
        switch to bools."""
        out = mo.zero_one_for_jumps_base(np.array([0.1, 0.9]), 0.5)
        assert [type(v) for v in out] == [int, int]


class TestSliders:
    def test_slices_slider(self):
        out = list(mo.slices_slider(np.arange(5), 2))
        assert [s.tolist() for s in out] == [[0, 1], [1, 2], [2, 3], [3, 4]]

    def test_slices_slider_full_length(self):
        out = list(mo.slices_slider(np.arange(4), 4))
        assert [s.tolist() for s in out] == [[0, 1, 2, 3]]

    def test_slices_jump_slider_full_windows(self):
        out = list(mo.slices_jump_slider(np.arange(10), 3, 3))
        assert [s.tolist() for s in out[:3]] == [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]

    def test_slices_jump_slider_no_partial_windows(self):
        out = list(mo.slices_jump_slider(np.arange(10), 3, 3))
        assert [s.tolist() for s in out] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


class TestRangesOf:
    @pytest.mark.parametrize(
        "array, expected",
        [
            ([0, 1, 1, 0, 0, 1, 1, 1, 0], [(0, 2), (4, 7)]),
            ([1, 1, 0, 0, 1], [(0, 1), (3, 4)]),
            ([0, 0, 0, 0], []),
        ],
    )
    def test_boundaries(self, array, expected):
        out = mo.ranges_of(np.array(array))
        assert isinstance(out, tuple)
        assert all(isinstance(r, mo.Range) for r in out)
        assert [tuple(r) for r in out] == expected


def _intersections_terminates(left, right, timeout=5):
    proc = multiprocessing.Process(target=mo.intersections, args=(left, right))
    proc.start()
    proc.join(timeout)
    alive = proc.is_alive()
    if alive:
        proc.terminate()
        proc.join()
    return not alive


class TestIntersections:
    def test_keeps_left_ranges_that_hit_right(self):
        left = [R(0, 2), R(5, 7), R(10, 12)]
        right = [R(1, 3), R(11, 15)]
        assert mo.intersections(left, right) == [R(0, 2), R(10, 12)]

    def test_empty(self):
        assert mo.intersections([], [R(0, 1)]) == []
        assert mo.intersections([R(0, 1)], []) == []

    def test_touching_ranges_terminate(self):
        assert _intersections_terminates([R(5, 9)], [R(0, 5)])


class TestRawOverlapPercent:
    def test_value(self):
        a = np.array([1, 1, 0, 0])
        b = np.array([1, 0, 1, 0])
        assert mo.raw_overlap_percent(a, b) == 0.25

    def test_bool_input(self):
        a = np.array([True, True, True, False])
        assert mo.raw_overlap_percent(a, a) == 0.75


class TestMerge:
    def test_sorted_merge(self):
        assert mo.merge([1, 4, 9], [2, 3, 10, 11]) == [1, 2, 3, 4, 9, 10, 11]

    def test_ties_take_right_first(self):
        assert mo.merge([1.0], [1]) == [1, 1.0]
        assert type(mo.merge([1.0], [1])[0]) is int

    def test_empty(self):
        assert mo.merge([], [1, 2]) == [1, 2]
        assert mo.merge([1, 2], []) == [1, 2]


class TestScaleArrays:
    def test_mapping(self):
        """Pinned as-is: min maps to `maximum` and max to `minimum`, so the
        output is flipped. Changing that must be a deliberate decision."""
        out = mo.scale_arrays(np.array([0.0, 5.0, 10.0]), 3, 0)
        np.testing.assert_allclose(out, [3.0, 1.5, 0.0])


class TestDelayEmbedding:
    def test_lag_1(self):
        out = mo.delay_embedding(np.arange(5), 3)
        np.testing.assert_array_equal(out, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])

    def test_lag_2(self):
        out = mo.delay_embedding(list(range(6)), 3, lag=2)
        np.testing.assert_array_equal(out, [[0, 2, 4], [1, 3, 5]])

    def test_too_short(self):
        with pytest.raises(ValueError, match="minimum required length is 5"):
            mo.delay_embedding(np.arange(4), 3, lag=2)


class TestSavePreprocessed:
    def test_creates_nested_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "x.npy"
        mo.save_preprocessed(np.arange(3), str(out), force=False)
        np.testing.assert_array_equal(np.load(out), [0, 1, 2])

    def test_existing_dir(self, tmp_path):
        out = tmp_path / "x.npy"
        mo.save_preprocessed(np.arange(3), str(out), force=False)
        assert out.is_file()

    def test_no_force_keeps_existing(self, tmp_path):
        out = tmp_path / "x.npy"
        mo.save_preprocessed(np.arange(3), str(out), force=False)
        mo.save_preprocessed(np.zeros(3), str(out), force=False)
        np.testing.assert_array_equal(np.load(out), [0, 1, 2])

    def test_force_overwrites(self, tmp_path):
        out = tmp_path / "x.npy"
        mo.save_preprocessed(np.arange(3), str(out), force=False)
        mo.save_preprocessed(np.zeros(3), str(out), force=True)
        np.testing.assert_array_equal(np.load(out), [0, 0, 0])

    def test_bare_filename(self, tmp_path, monkeypatch):
        """No directory part. The old os.makedirs('') crashed here."""
        monkeypatch.chdir(tmp_path)
        mo.save_preprocessed(np.arange(3), "x.npy", force=False)
        assert (tmp_path / "x.npy").is_file()


class TestDerivativeExact:
    def test_linear(self):
        out = mo.derivative(np.arange(10) * 2.0, fs=10)
        assert out.shape == (9,)
        np.testing.assert_allclose(out, 20.0)

    def test_window_nan_edges(self):
        out = mo.derivative(np.arange(10) * 2.0, fs=10, window_s=3)
        assert out.shape == (9,)
        assert np.isnan(out[[0, -1]]).all()
        np.testing.assert_allclose(out[1:-1], 20.0)


class TestBellCurve:
    def test_peak_and_width(self):
        x = np.array([1.0, 3.0, 5.0])
        out = mo.bell_curve(x, a=2.0, b=3.0, c=2.0)
        np.testing.assert_allclose(out, [2.0 / np.e, 2.0, 2.0 / np.e])


class TestRunningSmoother:
    def test_abs_window_and_zero_pad(self):
        array = np.tile([1.0, -1.0], 10)  # len 20 -> window 2
        out = mo.running_smoother(array)
        assert out.shape == (20,)
        np.testing.assert_allclose(out[:-1], 1.0)
        assert out[-1] == 0.0


class TestGetDictKeyWhereValue:
    def test_first_match(self):
        assert mo.get_dict_key_where_value({"a": 1, "b": 2, "c": 2}, 2) == "b"

    def test_missing(self):
        assert mo.get_dict_key_where_value({"a": 1}, 3) is None


if __name__ == "__main__":
    unittest.main()
