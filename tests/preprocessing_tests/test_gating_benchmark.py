"""Execution-time benchmarks for ``gating``, runnable under pytest.

Requires pytest-benchmark:  pip install pytest-benchmark

These are separated from the correctness suite (``test_gating.py``) so a normal
``pytest`` run stays fast. Run the benchmarks explicitly with:

    pytest test_gating_benchmark.py --benchmark-only

Useful invocations:
    # compare refactor vs original side by side, grouped per (size, method)
    pytest test_gating_benchmark.py --benchmark-only --benchmark-group-by=param:case

    # save a baseline, then later compare against it and fail on regression
    pytest test_gating_benchmark.py --benchmark-only --benchmark-save=main
    pytest test_gating_benchmark.py --benchmark-only --benchmark-compare=main \
           --benchmark-compare-fail=mean:10%
"""  # noqa: INP001

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from resurfemg.preprocessing.ecg_removal import gating_original, gating_vectorized

if TYPE_CHECKING:
    # Only needed for type checking; not imported at runtime so the file still
    # collects (and skips) cleanly when pytest-benchmark isn't installed.
    from collections.abc import Callable  # noqa: F401

    from pytest_benchmark.fixture import BenchmarkFixture

GATE_WIDTH: int = 205
METHODS: list[int] = [0, 1, 2, 3, 4]
METHOD_NAMES: dict[int, str] = {0: "zeros", 1: "interp", 2: "prior_mean", 3: "rms", 4: "quadratic"}
# (signal length, peak count): short clip, long recording, dense peaks
CONFIGS: list[tuple[int, int]] = [(60_000, 60), (500_000, 500), (500_000, 2_000)]


def _make(n: int, n_peaks: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    rng = np.random.default_rng(0)
    emg = rng.standard_normal(n)
    peaks = np.sort(rng.choice(np.arange(400, n - 400), n_peaks, replace=False))
    return emg, peaks


# Build the parameter matrix. `case` is a readable id shared by both the
# refactor and reference runs so --benchmark-group-by=param:case puts each
# matching pair in the same comparison group.
PARAMS = [
    pytest.param(n, npk, m, id=f"{n // 1000}k-{npk}pk-{METHOD_NAMES[m]}") for (n, npk) in CONFIGS for m in METHODS
]


IMPLS = {
    "original": gating_original,
    "refactor": gating_vectorized,
}


@pytest.mark.parametrize("impl_name", list(IMPLS))
@pytest.mark.parametrize(("n", "n_peaks", "method"), PARAMS)
def test_benchmark(
    benchmark: BenchmarkFixture,
    impl_name: str,
    n: int,
    n_peaks: int,
    method: int,
) -> None:
    impl = IMPLS[impl_name]
    emg, peaks = _make(n, n_peaks)
    benchmark.group = f"{n // 1000}k-{n_peaks}pk-{METHOD_NAMES[method]}"  # pair impls per case

    def run() -> npt.NDArray[np.float64]:
        return impl(emg.copy(), peaks, GATE_WIDTH, method)

    result = benchmark(run)
    assert result.shape == emg.shape
