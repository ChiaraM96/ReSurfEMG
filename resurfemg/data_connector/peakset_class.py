"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains data classes for standardized peak data storage and method
automation.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd

from resurfemg.postprocessing.event_detection import (
    onoffpeak_baseline_crossing,
    onoffpeak_slope_extrapolation,
)

numeric = int | float | np.number

array_like = np.ndarray | list | tuple


def _check_array_like(var: Any, var_name: str) -> np.ndarray:
    """Check if the variable is array-like and convert it to a numpy array.

    Args:
        var: The variable to check.
        var_name (str): The name of the variable for error messages.
    """
    if isinstance(var, array_like) and not isinstance(var, (str, bytes)):
        return np.asarray(var)
    msg = f"Invalid {var_name} type: '{type(var).__name__}'."
    raise TypeError(msg)


def _update_dataframe(
    dataframe_to_update: pd.DataFrame | None, new_dataframe: pd.DataFrame
) -> pd.DataFrame:
    if dataframe_to_update is not None:
        df_old = dataframe_to_update
        pre_existing_keys = list(set(new_dataframe.keys()) & set(df_old.keys()))
        df_old = df_old.drop(columns=pre_existing_keys)
        tests_df_merge = df_old.merge(new_dataframe, left_index=True, right_index=True)
        if not dataframe_to_update["peak_idx"].equals(tests_df_merge["peak_idx"]):
            msg = "Mismatched 'peak_idx' between old and new dataframes."
            raise ValueError(msg)
        return tests_df_merge
    return new_dataframe


class PeaksSet:
    """Data class to store, and process peak information."""

    def __init__(
        self,
        signal: array_like,
        t_data: array_like,
        peak_idxs: array_like | None = None,
    ):
        """Initialize the PeaksSet.

        Args:
            signal (np.ndarray): 1-dimensional signal data
            t_data (np.ndarray): time axis data
            peak_idxs (np.ndarray | list | None): Indices of peaks
        """
        self.t_data = _check_array_like(t_data, "t_data")
        self.signal = _check_array_like(signal, "signal")

        if peak_idxs is None:
            peak_idxs = np.array([])
        peak_idxs = _check_array_like(peak_idxs, "peak_idxs")

        self.peak_df = pd.DataFrame(data=peak_idxs, columns=pd.Index(["peak_idx"]))
        self.quality_values_df = pd.DataFrame(
            data=peak_idxs, columns=pd.Index(["peak_idx"])
        )
        self.quality_outcomes_df = pd.DataFrame(
            data=peak_idxs, columns=pd.Index(["peak_idx"])
        )
        self.time_products = None

    def __len__(self):
        return len(self.peak_df)

    def __getitem__(self, key: str):
        return self.peak_df[key].to_numpy()

    def __str__(self):
        return str(self.peak_df)

    def keys(self) -> pd.Index:
        """Return the column names of the peak DataFrame."""
        return self.peak_df.keys()

    def detect_on_offset(
        self,
        baseline: np.ndarray | None = None,
        method: Literal[
            "default", "baseline_crossing", "slope_extrapolation"
        ] = "default",
        fs: float | None = None,
        slope_window_s: int | None = None,
    ) -> None:
        """Detect the peak on- and offsets.

        See postprocessing.event_detection submodule.

        Args:
            baseline (np.ndarray | None): The baseline signal. (Default value = None)
            method (str): The method to use for detection. Possible values are
                "default", "baseline_crossing", and "slope_extrapolation".
                (Default value = "default")
            fs (float | None): The sampling rate. (Default value = None)
            slope_window_s (int | None): The window size for slope calculation.
                (Default value = None)

        Returns:
            None: None
        """
        if baseline is None:
            baseline = np.zeros(self.signal.shape)

        peak_idxs = self.peak_df["peak_idx"].to_numpy()

        match method:
            case "default" | "baseline_crossing":
                (start_idxs, end_idxs, _, _, valid_list) = onoffpeak_baseline_crossing(
                    self.signal, baseline, peak_idxs
                )

            case "slope_extrapolation":
                if fs is None:
                    msg = "Sampling rate is not defined."
                    raise ValueError(msg)

                if slope_window_s is None:
                    # TODO Insert valid default slope window
                    slope_window_s = int(fs // 5)

                (start_idxs, end_idxs, _, _, valid_list) = (
                    onoffpeak_slope_extrapolation(
                        self.signal, fs, peak_idxs, slope_window_s
                    )
                )
            case _:
                msg = "Detection algorithm does not exist."
                raise KeyError(msg)

        self.peak_df["start_idx"] = start_idxs
        self.peak_df["end_idx"] = end_idxs
        self.peak_df["valid"] = valid_list
        quality_outcomes_df = self.quality_outcomes_df
        quality_outcomes_df["baseline_detection"] = valid_list

        self.evaluate_validity(quality_outcomes_df)

    def update_test_outcomes(self, tests_df_new: pd.DataFrame) -> None:
        """Update the peak quality test outcomes.

        Add new peak quality test to self.quality_outcomes_df, and update
        existing entries.
        -----------------------------------------------------------------------

        Args:
          tests_df_new(pandas.DataFrame): Dataframe of test parameters per peak

        Returns:
          None: None

        """
        self.quality_values_df = _update_dataframe(self.quality_values_df, tests_df_new)

    def evaluate_validity(self, tests_df_new: pd.DataFrame) -> None:
        """Evaluate the validity of the provided peak quality test results.

        Update peak validity based on previously and newly executed tests
        in self.quality_outcomes_df.
        -----------------------------------------------------------------------

        Args:
          tests_df_new(pandas.DataFrame): Dataframe of passed tests per peak

        Returns:
          None: None

        """
        self.quality_outcomes_df = _update_dataframe(
            self.quality_outcomes_df, tests_df_new
        )
        outcomes = self.quality_outcomes_df.drop(columns="peak_idx")
        passed_tests = np.all(outcomes.to_numpy(), axis=1)
        self.peak_df["valid"] = passed_tests

    def sanitize(self) -> None:
        """Remove invalid peaks from the peak set.

        Delete invalid peak entries (self.peak_df['valid'] is False) from
        self.peak_df, self.quality_values_df, and self.quality_outcomes_df.
        """
        valid_idxs = self.peak_df["valid"].to_numpy()
        self.peak_df = self.peak_df.loc[valid_idxs].reset_index(drop=True)
        self.quality_outcomes_df = self.quality_outcomes_df.loc[valid_idxs].reset_index(
            drop=True
        )
        self.quality_values_df = self.quality_values_df.loc[valid_idxs].reset_index(
            drop=True
        )
