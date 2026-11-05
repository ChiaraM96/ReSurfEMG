"""Tests for resurfemg.helper_functions.visualization.

The functions both compute a spectrum and draw it, so the tests check the
numbers with synthetic sines (known answer), then check what was drawn on
the current axes. plt.show is patched out; the Agg backend never opens a
window anyway.
"""

import os
import unittest
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.fft import fft, fftfreq

from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.helper_functions import visualization as vis
from resurfemg.preprocessing.filtering import emg_bandpass_butter

matplotlib.use("Agg")

sample_emg = os.path.join(
    os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
    "test_data",
    "emg_data_synth_quiet_breathing.Poly5",
)

FS = 2048
F_SINE = 50.0
AMP = 3.0
N = 4 * FS  # 4 s, a whole number of sine cycles
T = np.arange(N) / FS
SINE = AMP * np.sin(2 * np.pi * F_SINE * T)


# --------------------------------------------------------------------------
# Real-data smoke tests (the original suite, with the slice fixed)
# --------------------------------------------------------------------------
class TestVisualizationMethods(unittest.TestCase):
    def setUp(self):
        sample_read = Poly5Reader(sample_emg)
        filtered = -emg_bandpass_butter(
            sample_read.samples, 1, 500, sample_read.sample_rate
        )
        # samples is (channels, samples): slice the time axis, not channels.
        self.sample_emg_filtered = filtered[:, : 30 * 2048]

    def tearDown(self):
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_show_power_spectrum(self, mock_show):
        power, freqs = vis.show_power_spectrum(
            self.sample_emg_filtered[0, :], 2048, 1024
        )
        self.assertEqual(len(freqs), 30 * 2048)
        self.assertEqual(len(power), 30 * 2048)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_show_psd_welch(self, mock_show):
        f, pxx_den = vis.show_psd_welch(
            self.sample_emg_filtered[0, :], 2048, 256, axis_spec=0
        )
        expected_length = 256 // 2 + 1
        self.assertEqual(len(f), expected_length)
        self.assertEqual(len(pxx_den), expected_length)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_show_periodogram(self, mock_show):
        f, pxx_den = vis.show_periodogram(
            self.sample_emg_filtered[0, :], 2048, 0
        )
        expected_length = 30 * 2048 // 2 + 1
        self.assertEqual(len(f), expected_length)
        self.assertEqual(len(pxx_den), expected_length)
        mock_show.assert_called_once()


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------
@pytest.fixture
def show(monkeypatch):
    """Patch plt.show, give each test a clean figure, close it after."""
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    plt.close("all")
    plt.figure()
    yield calls
    plt.close("all")


def _line():
    lines = plt.gca().get_lines()
    assert len(lines) == 1, f"expected one line, got {len(lines)}"
    return lines[0].get_xydata().T


CALLS = {
    "power_spectrum": lambda sig, **kw: vis.show_power_spectrum(
        sig, FS, FS / 2, **kw
    ),
    "welch": lambda sig, **kw: vis.show_psd_welch(sig, FS, 512, **kw),
    "periodogram": lambda sig, **kw: vis.show_periodogram(sig, FS, **kw),
}
TITLES = {
    "power_spectrum": "Power Spectrum",
    "welch": "Power Spectral Density",
    "periodogram": "Periodogram",
}


# --------------------------------------------------------------------------
# Plotting behaviour shared by all three functions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(CALLS))
class TestPlotting:
    @pytest.mark.parametrize(
        "axis_spec, scale",
        [(1, "log"), (0, "linear"), (True, "log"), (False, "linear")],
    )
    def test_axis_scale(self, show, name, axis_spec, scale):
        CALLS[name](SINE, axis_spec=axis_spec)
        assert plt.gca().get_yscale() == scale
        assert show == [1]

    def test_default_is_log(self, show, name):
        CALLS[name](SINE)
        assert plt.gca().get_yscale() == "log"

    def test_labels_and_title(self, show, name):
        CALLS[name](SINE, signal_unit="mV")
        ax = plt.gca()
        assert ax.get_xlabel() == "Frequency [Hz]"
        assert ax.get_ylabel() == "PSD [mV**2/Hz]"
        assert ax.get_title() == TITLES[name]

    @pytest.mark.parametrize("bad", [2, -1, "1", None, 0.5])
    def test_invalid_axis_spec(self, show, name, bad):
        with pytest.raises(ValueError, match="axis_spec"):
            CALLS[name](SINE, axis_spec=bad)
        assert show == []
        assert plt.gca().get_lines() == []

    def test_plotted_data_matches_return(self, show, name):
        out = CALLS[name](SINE, axis_spec=0)
        x, y = _line()
        if name == "power_spectrum":
            power, freqs = out
            keep = (freqs >= 0) & (freqs <= FS / 2)
            np.testing.assert_array_equal(x, freqs[keep])
            np.testing.assert_array_equal(y, power[keep])
        else:
            f, pxx = out
            np.testing.assert_array_equal(x, f)
            np.testing.assert_array_equal(y, pxx)


# --------------------------------------------------------------------------
# show_power_spectrum
# --------------------------------------------------------------------------
class TestShowPowerSpectrum:
    def test_returns_power_then_freqs(self, show):
        """Order is (power, freqs), the reverse of the other two functions.
        Pinned so a swap is a deliberate API change."""
        power, freqs = vis.show_power_spectrum(SINE, FS, 100)
        np.testing.assert_array_equal(freqs, fftfreq(N, 1 / FS))
        np.testing.assert_allclose(power, np.abs(fft(SINE)) ** 2)

    def test_full_length_including_negative_freqs(self, show):
        power, freqs = vis.show_power_spectrum(SINE, FS, 100)
        assert power.shape == freqs.shape == (N,)
        assert freqs.min() < 0

    def test_peak_at_sine_frequency(self, show):
        power, freqs = vis.show_power_spectrum(SINE, FS, 100)
        pos = freqs >= 0
        assert freqs[pos][np.argmax(power[pos])] == pytest.approx(F_SINE)

    @pytest.mark.parametrize("cutoff", [0, 49.75, 50, 100, FS])
    def test_plot_limited_to_cutoff_in_hz(self, show, cutoff):
        """t_window_s is a frequency cutoff in Hz (inclusive), despite the
        name. Negative frequencies are never plotted."""
        _, freqs = vis.show_power_spectrum(SINE, FS, cutoff, axis_spec=0)
        x, _ = _line()
        expected = freqs[(freqs >= 0) & (freqs <= cutoff)]
        np.testing.assert_array_equal(x, expected)
        assert x.min() == 0


# --------------------------------------------------------------------------
# show_psd_welch
# --------------------------------------------------------------------------
class TestShowPsdWelch:
    @pytest.mark.parametrize("nperseg", [256, 512, 1024])
    def test_length_follows_segment_samples(self, show, nperseg):
        f, pxx = vis.show_psd_welch(SINE, FS, nperseg)
        assert len(f) == len(pxx) == nperseg // 2 + 1
        assert f[-1] == pytest.approx(FS / 2)

    def test_peak_at_sine_frequency(self, show):
        f, pxx = vis.show_psd_welch(SINE, FS, 1024)
        assert f[np.argmax(pxx)] == pytest.approx(F_SINE, abs=FS / 1024)

    def test_power_is_conserved(self, show):
        """Integral of a density PSD = signal variance (A**2 / 2)."""
        f, pxx = vis.show_psd_welch(SINE, FS, 1024)
        power = np.sum(pxx) * (f[1] - f[0])
        assert power == pytest.approx(AMP**2 / 2, rel=0.02)

    def test_rejects_2d(self, show):
        with pytest.raises(ValueError, match="1-dimensional"):
            vis.show_psd_welch(np.vstack([SINE, SINE]), FS, 256)
        assert show == []


# --------------------------------------------------------------------------
# show_periodogram
# --------------------------------------------------------------------------
class TestShowPeriodogram:
    def test_length(self, show):
        f, pxx = vis.show_periodogram(SINE, FS)
        assert len(f) == len(pxx) == N // 2 + 1

    def test_peak_at_sine_frequency(self, show):
        f, pxx = vis.show_periodogram(SINE, FS)
        assert f[np.argmax(pxx)] == pytest.approx(F_SINE)

    def test_power_is_conserved(self, show):
        f, pxx = vis.show_periodogram(SINE, FS)
        power = np.sum(pxx) * (f[1] - f[0])
        assert power == pytest.approx(AMP**2 / 2, rel=1e-6)

    def test_axis_spec_positional(self, show):
        """Third positional argument is axis_spec, as in the real-data test."""
        vis.show_periodogram(SINE, FS, 0)
        assert plt.gca().get_yscale() == "linear"
