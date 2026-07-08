"""Parity and property tests for ``resurfemg.preprocessing.ecg_removal.gating``.

Strategy
--------
The refactored ``gating`` under test is compared, output-for-output, against a
faithful reproduction of the original algorithm (``gating_reference``). Any
divergence beyond floating-point tolerance is a behavior change and fails.

Coverage
--------
* parity across all four fill methods and a range of gate widths;
* scenarios that stress the parts this refactor historically got wrong:
  full-window gate coverage, gate overlap, signal edges, unsorted/duplicate
  peaks, and empty peak lists;
* invariants that hold regardless of the reference: the input is never
  mutated, the output shape/finiteness is sane, and empty peaks are a no-op.

Run with:  pytest test_gating.py -v
"""  # noqa: INP001

import numpy as np
import pytest

# --- locate the implementation under test and its RMS dependency ----------
# `gating` calls `evl.full_rolling_rms` internally; we reuse that exact
# routine so method-3 parity isn't polluted by a second RMS implementation.
from resurfemg.preprocessing.ecg_removal import gating_original, gating_vectorized

# ---------------------------------------------------------------------------
# Fixtures & scenario matrix
# ---------------------------------------------------------------------------
N = 20_000
ATOL = 1e-12
METHODS = [0, 1, 2, 3, 4]
GATE_WIDTHS = [205, 204, 100, 51]


@pytest.fixture(scope="module")
def signal() -> np.ndarray:
    """A deterministic, non-trivial EMG-like signal (drift + noise)."""
    rng = np.random.default_rng(20240607)
    x = rng.standard_normal(N).cumsum() * 0.01 + rng.standard_normal(N)
    return x.astype(float)


def _peaks(name: str) -> np.ndarray:
    """Named peak layouts. Kept as a function so each test gets a fresh array."""
    rng = np.random.default_rng(hash(name) % (2**32))
    layouts = {
        "sparse": np.sort(rng.choice(np.arange(400, N - 400), 80, replace=False)),
        "dense": np.sort(rng.choice(np.arange(300, N - 300), 400, replace=False)),
        "left_edge": np.array([50]),
        "right_edge": np.array([N - 50]),
        "both_edges": np.array([10, N - 10]),
        "overlap_pair": np.array([5000, 5050]),
        "overlap_triple": np.array([5000, 5040, 5080]),
        "adjacent": np.array([5000, 5204]),
        "unsorted_dup": np.array([9000, 300, 5000, 300, 12000]),
        "single_mid": np.array([N // 2]),
    }
    return layouts[name]


PEAK_LAYOUTS = [
    "sparse",
    "dense",
    "left_edge",
    "right_edge",
    "both_edges",
    "overlap_pair",
    "overlap_triple",
    "adjacent",
    "unsorted_dup",
    "single_mid",
]


# ---------------------------------------------------------------------------
# Parity: refactored output must equal the original algorithm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("layout", PEAK_LAYOUTS)
@pytest.mark.parametrize("gate_width", GATE_WIDTHS)
def test_parity_with_reference(signal: np.ndarray, method: int, layout: str, gate_width: int) -> None:
    peaks = _peaks(layout)

    # `gating` sorts internally; the reference is order-sensitive in overlap
    # regions, so feed it sorted peaks to compare like-for-like. This makes the
    # unsorted layout a real test that `gating` performs the sort.
    got = gating_vectorized(signal.copy(), peaks, gate_width, method)
    expected = gating_original(signal.copy(), np.sort(peaks).tolist(), gate_width, method)

    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=ATOL, equal_nan=True), (
        f"method={method} layout={layout} gate_width={gate_width}: "
        f"{int(np.sum(~np.isclose(got, expected, atol=ATOL, equal_nan=True)))} "
        f"samples differ (max |Δ|="
        f"{np.nanmax(np.abs(got - expected)):.3e})"
    )


# ---------------------------------------------------------------------------
# Coverage: method 0 must zero the whole gate window, not just start samples
# (guards the sparse-mask regression). Uses an all-ones signal so that any
# zero in the output can only come from gating.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gate_width", GATE_WIDTHS)
@pytest.mark.parametrize("layout", ["sparse", "overlap_pair", "left_edge", "right_edge"])
def test_method0_full_window_coverage(gate_width: int, layout: str) -> None:
    ones = np.ones(N, dtype=float)
    peaks = np.sort(_peaks(layout))
    half = gate_width // 2

    out = gating_vectorized(ones.copy(), peaks, gate_width, 0)

    expected_mask = np.zeros(N, dtype=bool)
    for p in peaks:
        expected_mask[max(0, p - half) : min(N, p + half)] = True

    zeroed = out == 0.0
    assert np.array_equal(zeroed, expected_mask), (
        f"gated coverage mismatch: zeroed {zeroed.sum()} samples, "
        f"expected {expected_mask.sum()} (a large under-count indicates the "
        f"start-only sparse-mask bug)"
    )


# ---------------------------------------------------------------------------
# Invariants (hold independent of the reference)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_empty_peaks_is_noop(signal: np.ndarray, method: int) -> None:
    """Empty peak_idxs must return the signal unchanged and must not raise."""
    out = gating_vectorized(signal.copy(), np.array([], dtype=int), 205, method)
    assert out.shape == signal.shape
    assert np.array_equal(out, signal)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("layout", ["sparse", "overlap_pair"])
def test_input_not_mutated(signal: np.ndarray, method: int, layout: str) -> None:
    """Gating must not modify the caller's array (it deep-copies)."""
    before = signal.copy()
    _ = gating_vectorized(signal, _peaks(layout), 205, method)
    assert np.array_equal(signal, before), "input signal was mutated in place"


@pytest.mark.parametrize("method", METHODS)
def test_unsorted_matches_sorted(signal: np.ndarray, method: int):
    """Result must not depend on the order peaks are supplied in."""
    peaks = _peaks("unsorted_dup")
    a = gating_vectorized(signal.copy(), peaks, 205, method)
    b = gating_vectorized(signal.copy(), np.sort(peaks), 205, method)
    assert np.allclose(a, b, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("method", METHODS)
def test_output_is_finite_where_signal_is(signal: np.ndarray, method: int) -> None:
    """Gating should not introduce NaN/inf into an otherwise finite signal."""
    out = gating_vectorized(signal.copy(), _peaks("sparse"), 205, method)
    assert np.all(np.isfinite(out))


def test_invalid_method_raises(signal: np.ndarray) -> None:
    with pytest.raises(ValueError):
        gating_vectorized(signal.copy(), _peaks("sparse"), 205, 99)
