"""sanity tests for the synthetic data functions"""

import numpy as np
import pytest
from scipy import signal as sps

import resurfemg.data_connector.synthetic_data as synth

FS_VENT = 100
FS_EMG = 2048
RR = 22
T_END = 30
IE_RATIO = 1 / 2
IE_FRACTION = IE_RATIO / (IE_RATIO + 1)
PEEP = 5
DP = 5
EMG_AMP = 5
EMG_SIM_PARAMETERS = {
    "ie_ratio": 1 / 2,  # ratio btw insp + expir phase
    "tau_mus_up": 0.3,
    "tau_mus_down": 0.3,
    "t_p_occs": [],
    "drift_amp": 100,
    "noise_amp": 2,
    "heart_rate": 80,
    "ecg_acceleration": 1.5,
    "ecg_amplitude": 200,
}

P_MUS_AMP = 5
VENT_SIM_PARAMETERS = {
    "ie_ratio": 1 / 2,  # ratio btw insp + expir phase
    "tau_mus_up": 0.3,
    "tau_mus_down": 0.3,
    "t_p_occs": [],
    "c": 0.050,
    "r": 5,
    "peep": 5,
    "flow_cycle": 0.25,  # Fraction F_max
    "flow_trigger": 2,  # L/min
    "tau_dp_up": 10,
    "tau_dp_down": 5,
}

LUNG_MECHANICS = {
    "c": 0.050,
    "r": 5,
}
VENT_SETTINGS = {
    "dp": DP,
    "peep": 5,
    "flow_cycle": 0.25,  # Fraction F_max
    "flow_trigger": 2,  # L/min
    "tau_dp_up": 10,
    "tau_dp_down": 5,
}


def _step(n_on: int, n_off: int = 0) -> np.ndarray:
    return np.concatenate([np.ones(n_on), np.zeros(n_off)])


def _breath_samples(fs: float, rr: float = RR) -> float:
    return fs * 60 / rr


def _p_mus(fs: int = FS_VENT, amp: float = 5, **pattern_kwargs) -> np.ndarray:
    pattern = synth.respiratory_pattern_generator(
        t_end=T_END, fs=fs, rr=RR, ie_ratio=IE_RATIO, **pattern_kwargs
    )
    return amp * synth.simulate_muscle_dynamics(pattern, fs=fs)


def _occlusion_mask(n: int, start: int, stop: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[start:stop] = True
    return mask


@pytest.fixture(autouse=True)
def _seed(monkeypatch):
    """Make the noise deterministic for both the legacy and Generator API."""
    np.random.seed(0)
    if hasattr(synth, "rng"):
        monkeypatch.setattr(synth, "rng", np.random.default_rng(0))

    # --- respiratory_pattern_generator ------------------------------------------


@pytest.fixture
def p_mus() -> np.ndarray:
    return _p_mus()


@pytest.fixture
def y_vent(p_mus) -> np.ndarray:
    return synth.simulate_ventilator_data(p_mus, fs_vent=FS_VENT)


@pytest.fixture
def activation() -> np.ndarray:
    return synth.simulate_muscle_dynamics(
        synth.respiratory_pattern_generator(t_end=T_END, fs=FS_EMG, rr=RR),
        fs=FS_EMG,
    )


@pytest.fixture
def pattern() -> np.ndarray:
    return synth.respiratory_pattern_generator(
        t_end=T_END, fs=FS_VENT, rr=RR, ie_ratio=IE_RATIO
    )


class TestSyntheticEmgData:
    respiratory_pattern = synth.respiratory_pattern_generator(
        t_end=T_END,
        fs=FS_EMG,
        rr=RR,
        ie_ratio=EMG_SIM_PARAMETERS["ie_ratio"],
        t_p_occs=EMG_SIM_PARAMETERS["t_p_occs"],
    )
    muscle_activation = synth.simulate_muscle_dynamics(
        block_pattern=respiratory_pattern,
        fs=FS_EMG,
        tau_mus_up=EMG_SIM_PARAMETERS["tau_mus_up"],
        tau_mus_down=EMG_SIM_PARAMETERS["tau_mus_down"],
    )
    emg_sim = synth.simulate_emg(
        muscle_activation=muscle_activation,
        fs_emg=FS_EMG,
        emg_amp=EMG_AMP,
        drift_amp=EMG_SIM_PARAMETERS["drift_amp"],
        noise_amp=EMG_SIM_PARAMETERS["noise_amp"],
    )

    def test_respiratory_pattern_generator(self):
        assert self.respiratory_pattern.shape[0] == FS_EMG * T_END

    def test_simulate_muscle_dynamics(self):
        assert self.muscle_activation.shape[0] == FS_EMG * T_END

    def test_simulate_emg(self):
        assert self.emg_sim.shape[0] == FS_EMG * T_END

    def test_emg_output_is_finite(self, activation):
        emg = synth.simulate_emg(activation, fs_emg=FS_EMG)
        assert emg.shape == activation.shape
        assert np.isfinite(emg).all()

    def test_emg_is_zero_without_activation_noise_or_drift(self):
        emg = synth.simulate_emg(
            np.zeros(T_END * FS_EMG), fs_emg=FS_EMG, drift_amp=0, noise_amp=0
        )
        np.testing.assert_array_equal(emg, 0)

    def test_emg_component_follows_activation(self, activation):
        emg = synth.simulate_emg(activation, fs_emg=FS_EMG, drift_amp=0, noise_amp=0)
        np.testing.assert_array_equal(emg[activation == 0], 0)
        assert np.std(emg[activation > 0.9]) > 0

    @pytest.mark.parametrize("amp", [1, 5, 20])
    def test_emg_rms_scales_with_emg_amp(self, amp):
        ones = np.ones(T_END * FS_EMG)
        emg = synth.simulate_emg(
            ones, fs_emg=FS_EMG, emg_amp=amp, drift_amp=0, noise_amp=0
        )
        ref = synth.simulate_emg(
            ones, fs_emg=FS_EMG, emg_amp=1, drift_amp=0, noise_amp=0
        )
        assert np.std(emg) / np.std(ref) == pytest.approx(amp, rel=0.02)

    @pytest.mark.parametrize("amp", [1, 2, 10])
    def test_emg_noise_scales_with_noise_amp(self, amp):
        zeros = np.zeros(T_END * FS_EMG)
        emg = synth.simulate_emg(zeros, fs_emg=FS_EMG, drift_amp=0, noise_amp=amp)
        ref = synth.simulate_emg(zeros, fs_emg=FS_EMG, drift_amp=0, noise_amp=1)
        assert np.std(emg) / np.std(ref) == pytest.approx(amp, rel=0.02)

    def test_emg_drift_is_low_frequency(self):
        drift = synth.simulate_emg(
            np.zeros(T_END * FS_EMG), fs_emg=FS_EMG, drift_amp=100, noise_amp=0
        )
        f, pxx = sps.welch(drift, fs=FS_EMG, nperseg=len(drift))
        assert pxx[f > 1].sum() / pxx.sum() < 0.01

    def test_emg_drift_scales_with_drift_amp(self):
        zeros = np.zeros(T_END * FS_EMG)
        np.random.seed(0)
        if hasattr(synth, "rng"):
            synth.rng = np.random.default_rng(0)
        small = synth.simulate_emg(zeros, fs_emg=FS_EMG, drift_amp=10, noise_amp=0)
        np.random.seed(0)
        if hasattr(synth, "rng"):
            synth.rng = np.random.default_rng(0)
        large = synth.simulate_emg(zeros, fs_emg=FS_EMG, drift_amp=100, noise_amp=0)
        np.testing.assert_allclose(large, 10 * small, atol=1e-6)


class TestSyntheticVentilatorData:
    respiratory_pattern = synth.respiratory_pattern_generator(
        t_end=T_END,
        fs=FS_VENT,
        rr=RR,
        ie_ratio=VENT_SIM_PARAMETERS["ie_ratio"],
        t_p_occs=VENT_SIM_PARAMETERS["t_p_occs"],
    )
    p_mus = P_MUS_AMP * synth.simulate_muscle_dynamics(
        block_pattern=respiratory_pattern,
        fs=FS_VENT,
        tau_mus_up=VENT_SIM_PARAMETERS["tau_mus_up"],
        tau_mus_down=VENT_SIM_PARAMETERS["tau_mus_down"],
    )
    t_occ_bool = np.zeros(p_mus.shape, dtype=bool)
    for t_occ in VENT_SIM_PARAMETERS["t_p_occs"]:
        t_occ_bool[
            int((t_occ - 1) * FS_VENT) : int((t_occ + 1 / RR * 60) * FS_VENT)
        ] = True

    y_vent = synth.simulate_ventilator_data(
        **{
            "p_mus": p_mus,
            "dp": DP,
            "fs_vent": FS_VENT,
            "t_occ_bool": t_occ_bool,
            **LUNG_MECHANICS,
            **VENT_SETTINGS,
        }
    )

    def test_respiratory_pattern_generator(self):
        assert self.respiratory_pattern.shape[0] == FS_VENT * T_END

    def test_simulate_muscle_dynamics(self):
        assert self.p_mus.shape[0] == FS_VENT * T_END

    def test_simulate_ventilator_data(self):
        assert self.y_vent.shape[0] == 3
        assert self.y_vent.shape[1] == FS_VENT * T_END

    def test_vent_output_is_finite(self, y_vent):
        assert np.isfinite(y_vent).all()

    def test_vent_channels_are_distinct(self, y_vent):
        _, flow, volume = y_vent
        assert not np.array_equal(flow, volume)

    def test_vent_volume_integrates_flow(self, y_vent):
        _, flow, volume = y_vent
        np.testing.assert_allclose(np.diff(volume), flow[1:] / FS_VENT, atol=1e-12)

    def test_vent_pressure_never_below_peep_without_occlusion(self, y_vent):
        assert y_vent[0].min() >= PEEP - 1e-9

    def test_vent_triggers_every_breath(self, y_vent):
        supported = y_vent[0] > PEEP + DP / 2
        n_triggers = np.count_nonzero(np.diff(supported.astype(int)) == 1)
        assert abs(n_triggers - T_END * RR / 60) <= 1

    @pytest.mark.parametrize("dp", [5, 10])
    def test_vent_peak_pressure_is_peep_plus_dp(self, p_mus, dp):
        y = synth.simulate_ventilator_data(p_mus, fs_vent=FS_VENT, dp=dp)
        assert y[0].max() == pytest.approx(PEEP + dp, abs=0.5)

    def test_vent_peep_only_shifts_pressure(self, p_mus, y_vent):
        y8 = synth.simulate_ventilator_data(p_mus, fs_vent=FS_VENT, peep=PEEP + 3)
        np.testing.assert_allclose(y8[0] - y_vent[0], 3)
        np.testing.assert_allclose(y8[1:], y_vent[1:])

    def test_vent_higher_compliance_gives_larger_volume(self, p_mus, y_vent):
        y = synth.simulate_ventilator_data(p_mus, fs_vent=FS_VENT, c=0.1)
        assert y[2].max() > y_vent[2].max()

    def test_vent_higher_resistance_gives_lower_peak_flow(self, p_mus, y_vent):
        y = synth.simulate_ventilator_data(p_mus, fs_vent=FS_VENT, r=10)
        assert y[1].max() < y_vent[1].max()

    def test_vent_none_occlusion_equals_all_false(self, p_mus, y_vent):
        y = synth.simulate_ventilator_data(
            p_mus, fs_vent=FS_VENT, t_occ_bool=np.zeros(len(p_mus), dtype=bool)
        )
        np.testing.assert_allclose(y, y_vent)

    def test_vent_unknown_kwarg_raises(self, p_mus):
        with pytest.raises(UserWarning, match="nope"):
            synth.simulate_ventilator_data(p_mus, fs_vent=FS_VENT, nope=1)

    def test_vent_does_not_modify_p_mus(self, p_mus):
        before = p_mus.copy()
        synth.simulate_ventilator_data(p_mus, fs_vent=FS_VENT)
        np.testing.assert_array_equal(p_mus, before)

    def test_vent_occlusion_zeroes_flow_and_volume(self, p_mus):
        occ = _occlusion_mask(len(p_mus), 10 * FS_VENT, 13 * FS_VENT)
        y = synth.simulate_ventilator_data(p_mus, fs_vent=FS_VENT, t_occ_bool=occ)
        assert np.all(y[1][occ] == 0)
        assert np.all(y[2][occ] == 0)

    def test_vent_occlusion_drops_pressure_below_peep(self, p_mus):
        occ = _occlusion_mask(len(p_mus), 10 * FS_VENT, 13 * FS_VENT)
        y = synth.simulate_ventilator_data(p_mus, fs_vent=FS_VENT, t_occ_bool=occ)
        assert y[0][occ].min() < PEEP - 1

    def test_vent_resumes_support_after_occlusion(self, p_mus):
        occ = _occlusion_mask(len(p_mus), 10 * FS_VENT, 13 * FS_VENT)
        y = synth.simulate_ventilator_data(p_mus, fs_vent=FS_VENT, t_occ_bool=occ)
        assert y[0][16 * FS_VENT :].max() > PEEP + DP / 2

    @pytest.mark.xfail(
        reason="ventilator simulation is unstable with occlusion in first half",
        strict=True,
    )
    def test_vent_occlusion_in_first_half_second_stays_finite(self, p_mus):
        occ = _occlusion_mask(len(p_mus), 10, 20)
        with np.errstate(all="ignore"):
            y = synth.simulate_ventilator_data(p_mus, fs_vent=FS_VENT, t_occ_bool=occ)
        assert np.isfinite(y).all()


class TestRespiratoryPatternGenerator:
    def test_pattern_is_binary(self, pattern):
        assert set(np.unique(pattern)) <= {0.0, 1.0}

    def test_pattern_duty_cycle_matches_ie_ratio(self, pattern):
        assert pattern.mean() == pytest.approx(IE_FRACTION, abs=0.01)

    @pytest.mark.parametrize("ie_ratio", [1 / 3, 1, 2])
    def test_pattern_duty_cycle_follows_ie_ratio(self, ie_ratio):
        p = synth.respiratory_pattern_generator(
            t_end=T_END, fs=FS_VENT, rr=RR, ie_ratio=ie_ratio
        )
        assert p.mean() == pytest.approx(ie_ratio / (ie_ratio + 1), abs=0.01)

    @pytest.mark.parametrize("rr", [12, 22, 30])
    def test_pattern_breath_count_follows_rr(self, rr):
        p = synth.respiratory_pattern_generator(t_end=T_END, fs=FS_VENT, rr=rr)
        n_onsets = np.count_nonzero(np.diff(p) > 0)
        assert abs(n_onsets - T_END * rr / 60) <= 1

    def test_pattern_none_occlusions_equals_empty(self, pattern):
        p_empty = synth.respiratory_pattern_generator(
            t_end=T_END, fs=FS_VENT, rr=RR, ie_ratio=IE_RATIO, t_p_occs=[]
        )
        np.testing.assert_array_equal(pattern, p_empty)

    @pytest.mark.parametrize("fs", [FS_VENT, FS_EMG])
    @pytest.mark.parametrize("t_p_occ", [10.0, 12.3])
    def test_occlusion_inserts_one_breath_at_breath_start(self, fs, t_p_occ):
        p = synth.respiratory_pattern_generator(
            t_end=T_END, fs=fs, rr=RR, ie_ratio=IE_RATIO, t_p_occs=[t_p_occ]
        )
        t_occ = np.floor(t_p_occ * RR / 60) * 60 / RR
        i_occ = int(t_occ * fs)
        n_breath = int(_breath_samples(fs)) + 1
        expected = (np.arange(n_breath) / _breath_samples(fs)) % 1 < IE_FRACTION

        segment = p[i_occ : i_occ + n_breath]
        # Allow one sample of rounding at the on/off edge.
        assert np.count_nonzero(segment != expected) <= 2

    def test_occlusion_only_changes_one_breath(self):
        base = synth.respiratory_pattern_generator(t_end=T_END, fs=FS_VENT, rr=RR)
        occ = synth.respiratory_pattern_generator(
            t_end=T_END, fs=FS_VENT, rr=RR, t_p_occs=[10.0]
        )
        i_occ = int(np.floor(10.0 * RR / 60) * 60 / RR * FS_VENT)
        n_breath = int(_breath_samples(FS_VENT)) + 1
        outside = np.ones(len(base), dtype=bool)
        outside[i_occ : i_occ + n_breath] = False
        np.testing.assert_array_equal(occ[outside], base[outside])

    def test_multiple_occlusions(self):
        occ = synth.respiratory_pattern_generator(
            t_end=T_END, fs=FS_VENT, rr=RR, t_p_occs=[5.0, 15.0, 25.0]
        )
        assert len(occ) == T_END * FS_VENT
        assert set(np.unique(occ)) <= {0.0, 1.0}

    def test_occlusion_near_end_is_truncated(self, capsys):
        # With t_end=29 the last breath (27.3-30.0 s) can't fit.
        p = synth.respiratory_pattern_generator(
            t_end=29, fs=FS_VENT, rr=RR, t_p_occs=[28]
        )
        assert len(p) == 29 * FS_VENT
        assert "full respiratory cycle" in capsys.readouterr().out

    def test_occlusion_far_from_end_is_silent(self, capsys):
        synth.respiratory_pattern_generator(
            t_end=T_END, fs=FS_VENT, rr=RR, t_p_occs=[5.0]
        )
        assert capsys.readouterr().out == ""

    def test_muscle_starts_at_zero_and_stays_bounded(self, pattern):
        act = synth.simulate_muscle_dynamics(pattern, fs=FS_VENT)
        assert act[0] == 0
        assert act.min() >= 0
        assert act.max() <= 1

    @pytest.mark.parametrize("tau", [0.1, 0.3, 1.0])
    def test_muscle_rise_is_first_order(self, tau):
        n_tau = int(tau * FS_VENT)
        k = np.arange(5 * n_tau)
        act = synth.simulate_muscle_dynamics(_step(len(k)), fs=FS_VENT, tau_mus_up=tau)
        np.testing.assert_allclose(act, 1 - (1 - 1 / n_tau) ** k)
        # Roughly 1 - 1/e after one tau (the discrete step makes it a bit higher).
        assert act[n_tau] == pytest.approx(1 - np.exp(-1), abs=0.03)

    @pytest.mark.parametrize("tau", [0.1, 0.3, 1.0])
    def test_muscle_decay_is_first_order(self, tau):
        n_on = 20 * FS_VENT  # long enough to saturate
        n_tau = int(tau * FS_VENT)
        k = np.arange(5 * n_tau)
        act = synth.simulate_muscle_dynamics(
            _step(n_on, len(k)), fs=FS_VENT, tau_mus_up=0.05, tau_mus_down=tau
        )
        np.testing.assert_allclose(act[n_on:], act[n_on] * (1 - 1 / n_tau) ** k)

    def test_muscle_up_and_down_taus_are_independent(self):
        step = _step(2 * FS_VENT, 2 * FS_VENT)
        fast_down = synth.simulate_muscle_dynamics(
            step, fs=FS_VENT, tau_mus_up=0.3, tau_mus_down=0.05
        )
        slow_down = synth.simulate_muscle_dynamics(
            step, fs=FS_VENT, tau_mus_up=0.3, tau_mus_down=1.0
        )
        n_on = 2 * FS_VENT
        np.testing.assert_allclose(fast_down[: n_on + 1], slow_down[: n_on + 1])
        assert fast_down[n_on + 30] < slow_down[n_on + 30]

    @pytest.mark.parametrize(
        "convert",
        [
            lambda x: x.astype(int),
            lambda x: x.astype(bool),
            lambda x: x.astype(int).tolist(),
        ],
        ids=["int_array", "bool_array", "int_list"],
    )
    def test_muscle_non_float_input_matches_float(self, pattern, convert):
        ref = synth.simulate_muscle_dynamics(pattern, fs=FS_VENT)
        act = synth.simulate_muscle_dynamics(convert(pattern), fs=FS_VENT)
        assert np.issubdtype(np.asarray(act).dtype, np.floating)
        np.testing.assert_allclose(act, ref)
