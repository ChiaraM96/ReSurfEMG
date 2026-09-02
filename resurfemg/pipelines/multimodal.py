# Custom code libraries from ReSurfEMG  # noqa: CPY001
import importlib
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resurfemg.data_connector.data_classes import (
    EmgDataGroup,
    VentilatorDataGroup,
)
from resurfemg.legacy import helper_functions as legacy
from resurfemg.postprocessing import features as feat

_available_methods = [
    "gating",
    "emg_bandpass_butter_sample",
    "full_rolling_rms",
    "snr_pseudo",
]


class MultimodalDataGroup:
    """Class for handling VentiltorDataGroup and EmgDataGroup objects together."""

    def __init__(
        self,
        vent_timeseries: VentilatorDataGroup,
        emg_timeseries: EmgDataGroup,
        use_legacy: bool = False,
    ):
        self.vent_timeseries: VentilatorDataGroup = vent_timeseries
        self.emg_timeseries: EmgDataGroup = emg_timeseries
        self.use_legacy: bool = use_legacy

    def _legacy_adapter(
        self, method: str, channel_idxs: list[int] | None = None, **kwargs
    ) -> dict[int, np.ndarray]:
        if channel_idxs is None:
            channel_idxs = [0]
        if method not in _available_methods:
            msg = "Invalid method"
            raise ValueError(msg)

        _kwargs = kwargs.copy()
        fn = getattr(legacy, method)
        return {i: fn(**_kwargs) for i in channel_idxs}

    def set_labels(self) -> None:
        """Set the labels for the ventilator and EMG timeseries."""
        self.v_vent_idx = next(
            i for i, s in enumerate(self.vent_timeseries.labels) if "V" in s
        )
        self.p_vent_idx = next(
            i for i, s in enumerate(self.vent_timeseries.labels) if "P" in s
        )
        self.f_idx = next(
            i for i, s in enumerate(self.vent_timeseries.labels) if "F" in s
        )

        self.emg_timeseries.labels = ["ECG", "EMGdi", "EMGic"]
        for idx, label in enumerate(self.emg_timeseries.labels):
            self.emg_timeseries[idx].label = label

        self.emg_timeseries.ecg_idx = next(
            i for i, s in enumerate(self.emg_timeseries.labels) if "ECG" in s
        )
        self.emg_idx = [
            i
            for i in range(len(self.emg_timeseries.channels))
            if i != self.emg_timeseries.ecg_idx
        ]
        self.ecg_idx = self.emg_timeseries.ecg_idx

    def calculate_peak_values(self, trial: Literal["PEEP", "PS"] = "PEEP") -> None:
        """Calculate the peak values for the EMG and ventilator signals.

        Calculates the Tidal Volume (TV) for the ventilator signal
        and the sEA for the EMG signals.

        Args:
            trial (Literal["PEEP", "PS"]): The type of trial. Defaults to "PEEP".
        """
        if trial == "PEEP":
            emg_peak_set_name = "emg_breaths"
            vent_peak_set_name = "Pocc"
            v_idx = self.p_vent_idx
            v_measure = "Pocc"
        else:
            emg_peak_set_name = "neural_breaths"
            vent_peak_set_name = "neural_breaths"
            v_idx = self.v_vent_idx
            v_measure = "TV"

        for idx in self.emg_idx:
            _amp = feat.amplitude(
                signal=np.asarray(self.emg_timeseries[idx]["env"]),
                peak_idxs=self.emg_timeseries[idx]
                .peaks[emg_peak_set_name]
                .peak_df["peak_idx"]
                .to_numpy(),
                baseline=np.asarray(self.emg_timeseries[idx]["baseline"]),
            )
            self.emg_timeseries[idx].peaks[emg_peak_set_name].peak_df["sEA"] = _amp

        _amp = feat.amplitude(
            signal=np.asarray(self.vent_timeseries[v_idx]["raw"]),
            peak_idxs=self.vent_timeseries[v_idx]
            .peaks[vent_peak_set_name]
            .peak_df["peak_idx"]
            .to_numpy(),
            baseline=np.asarray(self.vent_timeseries[v_idx]["baseline"]),
        )
        self.vent_timeseries[v_idx].peaks[vent_peak_set_name].peak_df[v_measure] = _amp

    def calculate_PTP(self) -> None:  # noqa: N802
        """Calculate the pressure-time product (PTP) for Pocc peaks."""
        # calculate PTPocc as the area between Paw and the moving baseline
        # between the on- and offset of the Pocc peak
        self.vent_timeseries[self.p_vent_idx].calculate_time_products(
            peak_set_name="Pocc",
            parameter_name="PTPocc",
            aub_reference_signal=self.vent_timeseries[self.p_vent_idx]["baseline"],
            include_aub=True,
        )

    def process_ventilator_breaths(self, trial: Literal["PEEP", "PS"] = "PEEP") -> None:
        """Process the ventilator breaths and Pocc peaks.

        Operates on the volume time series for both PEEP and PS trials,
        and on the pressure time series for PEEP trials only.

        Calculates the baseline for the ventilator signals, detects the ventilator peaks
        (and occluded breaths for PEEP trials) and calculates their on- and offsets.

        Args:
            trial (Literal["PEEP", "PS"]): The type of trial. Defaults to "PEEP".
        """
        if trial == "PEEP":
            vent_idx = [self.v_vent_idx, self.p_vent_idx]
            peak_set_name = ["ventilator_breaths", "Pocc"]
        else:
            vent_idx = [self.v_vent_idx]
            peak_set_name = ["ventilator_breaths"]
        for _vent_idx in vent_idx:
            self.vent_timeseries.baseline(
                channel_idxs=[_vent_idx], signal_io=("raw", "baseline")
            )
        self.vent_timeseries.find_ventilator_peaks(
            channel_io=(self.v_vent_idx, self.v_vent_idx),
            overwrite=True,
            peak_set_name="ventilator_breaths",
        )
        if trial == "PEEP":
            self.vent_timeseries.find_occluded_breaths()

        for _vent_idx, _peak_set_name in zip(vent_idx, peak_set_name, strict=False):
            self.vent_timeseries[_vent_idx].peaks[_peak_set_name].detect_on_offset(
                baseline=self.vent_timeseries[_vent_idx]["baseline"],
                fs=self.vent_timeseries.fs,
            )

    def process_emg(self) -> None:
        """Process the EMG signals.

        This method filters the EMG signals, detects the QRS complexes
        inin the ECG signal, performs gating to remove ECG artifacts from
        the EMG signals, computes the sEA envelope, and calculates the
        improved moving slopesum baseline for the EMG signals.
        """
        # 4. Processing of sEMG signals
        ## QRS complex detection
        # filter the data
        self.emg_timeseries.filter()

        # get the indexes of the ecg and emg channels

        # Filter the ECG signal and detect the QRS complexes
        self.emg_timeseries.get_ecg_peaks(name="ecg")

        if self.use_legacy:
            env_window = int(0.2 * self.emg_timeseries.param["fs"])
            gate_peaks = (
                self.emg_timeseries[self.ecg_idx]
                .peaks["ecg"]
                .peak_df["peak_idx"]
                .to_numpy()
            )
            for emg_idx in self.emg_idx:
                self.emg_timeseries[emg_idx]["gated"] = legacy.gating(
                    src_signal=self.emg_timeseries[emg_idx]["filt"],
                    gate_peaks=gate_peaks,
                    gate_width=int(0.1 * self.emg_timeseries.param["fs"]),
                    method=1,
                )
                self.emg_timeseries[emg_idx]["env"] = legacy.full_rolling_rms(
                    data_emg=np.asarray(self.emg_timeseries[emg_idx]["gated"]),
                    window_length=env_window,
                )
        else:
            self.emg_timeseries.gating(
                channel_idxs=self.emg_idx,
                ecg_peakset_name="ecg",
                gate_width_samples=int(0.1 * self.emg_timeseries.param["fs"]),
            )
            # compute the sEA envelope
            self.emg_timeseries.envelope(
                channel_idxs=self.emg_idx,
                env_window=int(0.2 * self.emg_timeseries.param["fs"]),
                env_type="rms",
            )
        ## Improved moving slopesum baseline
        # calculate the slopesum moving baseline
        # if self.baseline_type == "slopesum":
        self.emg_timeseries.baseline(
            channel_idxs=self.emg_idx,
            base_method="slopesum_baseline",
            window_s=int(7.5 * self.emg_timeseries.param["fs"]),
            ma_window=int(0.5 * self.emg_timeseries.param["fs"]),
            signal_io=("env", "baseline_slopesum"),
        )
        # else:
        self.emg_timeseries.baseline(
            channel_idxs=self.emg_idx, signal_io=("env", "baseline")
        )

    def link_breaths(self, trial: Literal["PEEP", "PS"] = "PEEP") -> None:
        """Link the detected breaths across modalities.

        For PEEP trials, links the EMG peaks to the closest Pocc peaks.
        For PS trials, links the ventilator peaks to the closest neural peaks.

        Args:
            trial (Literal["PEEP", "PS"]): The type of trial. Defaults to "PEEP".
        """
        if trial == "PEEP":
            for idx in self.emg_idx:
                # link EMG peaks to the closest Pocc peaks
                self.emg_timeseries[idx].link_peak_set(
                    peak_set_name="emg_breaths",
                    t_reference_peaks=self.vent_timeseries[
                        self.vent_timeseries.p_vent_idx
                    ]
                    .peaks["Pocc"]
                    .peak_df["peak_idx"]
                    .to_numpy()
                    / self.vent_timeseries.param["fs"],
                    linked_peak_set_name="Pocc",
                )
        else:
            for idx in self.emg_idx:
                # link tidal peaks to the closest neural peaks
                self.vent_timeseries[self.v_vent_idx].link_peak_set(
                    peak_set_name="ventilator_breaths",
                    t_reference_peaks=self.emg_timeseries[idx]
                    .peaks["neural_breaths"]
                    .peak_df["peak_idx"]
                    .to_numpy()
                    / self.emg_timeseries.param["fs"],
                    linked_peak_set_name="neural_breaths",
                )

    def get_emg_breaths(self, trial: Literal["PEEP", "PS"] = "PEEP") -> None:
        """Process the EMG breaths.

        Detects peaks in the EMG signals and calculates their onsets and offsets.

        Args:
            trial (Literal["PEEP", "PS"]): The type of trial. Defaults to "PEEP".
        """
        ## Peak detection
        # detect the sEA peaks
        emg_peak_set_name = "emg_breaths" if trial == "PEEP" else "neural_breaths"
        self.emg_timeseries.detect_emg_breaths(
            channel_idxs=self.emg_idx,
            peak_set_name=emg_peak_set_name,
            signal_io=(("env", "baseline"),),
            # prominence_factor=0.1,
            # min_peak_width_s=int(0.2 * self.emg_timeseries.param["fs"]),
        )

    def detect_emg_on_offsets(self, trial: Literal["PEEP", "PS"] = "PEEP") -> None:
        """Process the EMG breaths.

        Detects peaks in the EMG signals and calculates their onsets and offsets.

        Args:
            trial (Literal["PEEP", "PS"]): The type of trial. Defaults to "PEEP".
        """

        def _compute_legacy_baseline(idx: int, peakset_name: str) -> np.ndarray:
            """Compute the legacy baseline for the EMG signals.

            This method calculates the legacy baseline for the EMG signals
            using a moving baseline approach.

            Args:
                idx (int): Index of the EMG channel.

            Returns:
                numpy.ndarray: The legacy baseline for the EMG signal.
            """
            w = int(7.5 * self.emg_timeseries.param["fs"])
            ts = self.emg_timeseries[idx]
            env = np.asarray(ts["env"])
            base = np.asarray(ts["baseline"])  # plain moving, 7.5 s
            s = pd.Series(base)
            std = s.rolling(w, min_periods=1, center=True).std().to_numpy()
            mean = s.rolling(w, min_periods=1, center=True).mean().to_numpy()
            pk = ts.peaks[peakset_name]["peak_idx"]  # LINKED occlusion peaks
            return base * (1 + np.nanmedian(env[pk]) * std / mean**2)

        ## Peak detection
        # detect the sEA peaks
        emg_peak_set_name = "Pocc" if trial == "PEEP" else "neural_breaths"

        # get the on and offsets as baseline crossing points
        for idx in self.emg_idx:
            self.emg_timeseries[idx].peaks[emg_peak_set_name].detect_on_offset(
                baseline=_compute_legacy_baseline(idx, emg_peak_set_name)
                if self.use_legacy
                else self.emg_timeseries[idx]["baseline_slopesum"],
                fs=self.emg_timeseries.fs,
            )

    def plot_data(
        self, trial: Literal["PEEP", "PS"] = "PEEP", plot_markers: bool = True
    ) -> np.ndarray:
        """Plot the EMG and ventilator data with detected peaks and markers.

        Args:
            trial (Literal["PEEP", "PS"]): The type of trial. Defaults to "PEEP".
            plot_markers (bool): Whether to plot markers for detected peaks.
                Defaults to True.

        Returns:
            np.ndarray: The axes of the plotted figures.
        """
        # plot the detected Pocc peaks and the linked EMG peaks
        peak_set_name: list[str] = []
        if plot_markers is True:
            if trial == "PS":
                peak_set_name = ["neural_breaths", "neural_breaths", "neural_breaths"]
            else:
                peak_set_name = ["Pocc", "Pocc", "Pocc"]
        nrows = 3
        ncols = 2
        axes = plt.subplots(
            nrows=nrows, ncols=ncols, sharex=True, figsize=(ncols * 6, nrows * 2)
        )[1]

        axes_emg = axes[:, 0]
        axes_vent = axes[:, 1]

        self.emg_timeseries.plot_full(
            axes=axes_emg,
            baseline_bool=True,
        )

        self.vent_timeseries.plot_full(
            axes=axes_vent,
            baseline_bool=True,
        )

        if plot_markers is True:
            self.emg_timeseries.plot_markers(
                peak_set_name=peak_set_name,
                axes=np.transpose([axes_emg]),
                colors=["tab:red", "yellow", "yellow"],
                valid_only=False,
            )
            self.vent_timeseries.plot_markers(
                peak_set_name=peak_set_name,
                axes=np.transpose([axes_vent]),
                colors=["tab:red", "yellow", "yellow"],
                valid_only=False,
            )
            try:
                self.emg_timeseries.plot_markers(
                    peak_set_name=peak_set_name,
                    axes=np.transpose([axes_emg]),
                    colors=["tab:cyan", "tab:green", "tab:green"],
                    valid_only=True,
                )
            except:
                print("No valid EMG peaks detected.")
            try:
                self.vent_timeseries.plot_markers(
                    peak_set_name=peak_set_name,
                    axes=np.transpose([axes_vent]),
                    colors=["tab:cyan", "tab:green", "tab:green"],
                    valid_only=True,
                )
            except:
                print("No valid Ventilator peaks detected.")

        axes_emg[0].set_title("EMG data")
        axes_emg[-1].set_xlabel("t (s)")
        axes_vent[0].set_title("Ventilator data")
        axes_vent[-1].set_xlabel("t (s)")
        plt.show()
        return axes

    def plot_peaks(self, trial: Literal["PEEP", "PS"] = "PEEP") -> None:
        """Plot the detected peaks for the EMG and ventilator signals.

        Args:
            trial (Literal["PEEP", "PS"]): The type of trial. Defaults to "PEEP".
        """
        peak_set_name = "Pocc" if trial == "PEEP" else "neural_breaths"
        n_peaks = len(
            self.vent_timeseries[self.p_vent_idx]
            .peaks[peak_set_name]
            .peak_df["start_idx"]
            .to_numpy()
        )
        vent_idx = [self.p_vent_idx, self.v_vent_idx]
        n_peaks_to_plot = min(10, n_peaks)
        if n_peaks:
            fig, axes = plt.subplots(
                nrows=4,
                ncols=n_peaks_to_plot,
                sharey="row",
                sharex="col",
                figsize=(6 * n_peaks_to_plot, 6),
                layout="constrained",
            )
            axes_emg = axes[:2, :] if n_peaks_to_plot > 1 else axes[:2]
            axes_vent = axes[2:, :] if n_peaks_to_plot > 1 else axes[2:]
            colors = ["tab:cyan", "tab:orange", "tab:red"]
            margin_s = 5  # margin in seconds around the peak to plot

            self.emg_timeseries.plot_peaks(
                channel_idxs=self.emg_idx,
                peak_set_name=[peak_set_name],
                axes=axes_emg,
                colors=colors,
                margin_s=margin_s * self.emg_timeseries.param["fs"],
                n_peaks_to_plot=n_peaks_to_plot,
            )

            self.emg_timeseries.plot_markers(
                channel_idxs=self.emg_idx,
                peak_set_name=[peak_set_name],
                axes=axes_emg,
                colors=colors,
                n_peaks_to_plot=n_peaks_to_plot,
            )

            for i in range(len(self.emg_idx)):
                axes_emg[i, 0].set_ylabel(self.emg_timeseries.labels[self.emg_idx[i]])

            self.vent_timeseries.plot_peaks(
                channel_idxs=vent_idx,
                peak_set_name=[peak_set_name],
                axes=axes_vent,
                colors=colors,
                margin_s=margin_s * self.vent_timeseries.param["fs"],
                n_peaks_to_plot=n_peaks_to_plot,
            )

            self.vent_timeseries.plot_markers(
                channel_idxs=vent_idx,
                peak_set_name=[peak_set_name],
                axes=axes_vent,
                colors=colors,
                n_peaks_to_plot=n_peaks_to_plot,
            )

            for i in range(len(vent_idx)):
                axes_vent[i, 0].set_ylabel(self.vent_timeseries.labels[vent_idx[i]])

            for i in range(n_peaks_to_plot):
                t_peak = (
                    self.vent_timeseries[self.vent_timeseries.p_vent_idx]
                    .peaks[peak_set_name]
                    .peak_df["peak_idx"]
                    .to_numpy()[i]
                    / self.vent_timeseries.param["fs"]
                )
                axes_emg[0, i].set_title(f"Peak {i + 1}\nt = {t_peak:.2f} s")

            fig.canvas.header_visible = False  # pyright: ignore[reportAttributeAccessIssue]
            fig.canvas.toolbar_visible = True  # pyright: ignore[reportAttributeAccessIssue]
            plt.show()

    def calculate_ETP(self, trial: Literal["PEEP", "PS"] = "PEEP") -> None:  # noqa: N802
        """Calculate the expiratory time product (ETP) for the EMG signals.

        Args:
            trial (Literal["PEEP", "PS"]): The type of trial. Defaults to "PEEP".
        """
        peak_set_name = "Pocc" if trial == "PEEP" else "neural_breaths"
        ## ETP calculation
        self.emg_timeseries.calculate_time_products(
            channel_idxs=self.emg_idx,
            peak_set_name=peak_set_name,
            include_aub=True,
            aub_window_s=5 * self.emg_timeseries.param["fs"],
            parameter_name="ETP",
        )

    def compute_quality_criteria(self, trial: Literal["PEEP", "PS"] = "PEEP") -> None:
        """Compute quality criteria for the EMG and ventilator signals.

        Args:
            trial (Literal["PEEP", "PS"]): The type of trial. Defaults to "PEEP".
        """
        ## Quality criteria
        if trial == "PEEP":
            peak_set_name = "Pocc"
            vent_idx = self.p_vent_idx
            vent_parameter_name = "PTPocc"
            skip_tests = None
        else:
            peak_set_name = "neural_breaths"
            vent_idx = self.v_vent_idx
            vent_parameter_name = "PTP_neural_breaths"
            skip_tests = ["relative_aub", "relative_etp"]
        for idx in self.emg_idx:
            if not self.emg_timeseries[idx].peaks[peak_set_name].peak_df.empty:
                self.emg_timeseries[idx].test_emg_quality(
                    peak_set_name=peak_set_name,
                    parameter_names={"time_product": "ETP"},
                    verbose=False,
                    skip_tests=skip_tests,
                )
        if trial == "PEEP":
            self.vent_timeseries[vent_idx].test_pocc_quality(
                peak_set_name=peak_set_name,
                parameter_names={"time_product": vent_parameter_name},
                skip_tests=["consecutive_poccs"],
                verbose=False,
            )
