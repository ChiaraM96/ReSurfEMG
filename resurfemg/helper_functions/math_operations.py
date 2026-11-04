"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions for math operations to support the functions in
ReSurfEMG
"""  # noqa: D205

import contextlib
from collections.abc import Generator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd


class Range(NamedTuple):
    """Utility class for working with ranges (intervals).

    Attributes
    ----------
    start : int
        Start of the range
    end : int
        End of the range

    """

    start: int
    end: int

    def intersects(self, other: "Range") -> bool:
        """Check if this range intersects another range.

        Returns `True` if this range intersects `other` range.


        Parameters
        ----------
        other : resurfemg.helper_functions.Range
            Another range to compare this one to


        Returns
        -------
        bool
            True` if this range intersects another range

        """
        return (
            ((self.end >= other.end) and (self.start < other.end))
            or ((self.end >= other.start) and (self.start < other.start))
            or ((self.end < other.end) and (self.start >= other.start))
        )

    def precedes(self, other: "Range") -> bool:
        """Check if this range precedes another range.

        Returns `True` if this range precedes `other` range.

        Parameters
        ----------
        other : resurfemg.helper_functions.Range
            Another range to compare this one to


        Returns
        -------
        bool
            True` if this range strictly precedes another range

        """
        return self.end < other.start

    def to_slice(self) -> slice:
        """Convert this range to a :class:`slice`.

        Returns
        -------
        slice
            A slice with its start set to this range's start and end set
            to this range's end

        """
        return slice(*map(int, self))  # maps whole tuple set


def zero_one_for_jumps_base(array: np.ndarray, cut_off: float) -> list:
    """Make an array binary (0, 1) based on a cut-off value.

    This function takes an array and makes it binary (0, 1) based
    on a cut-off value.


    Parameters
    ----------
    array : numpy.ndarray
        An array
    cut_off : float
        The number defining a cut-off line for binarization


    Returns
    -------
    list
        A list of 0s and 1s representing the binarized array.

    """
    return [int(i >= cut_off) for i in array]


def slices_slider(
    array_sample: np.ndarray, slice_len: int
) -> Generator[np.ndarray, None, None]:
    """Produce continuous sequential slices of an array.

    This function produces continuous sequential slices over an
    array of a certain length.  The inputs are the following -
    `array_sample`, the signal and `slice_len` - the
    window which you wish to slide with. The function yields, does
    not return these slices.


    Parameters
    ----------
    array_sample : numpy.ndarray
        array containing the signal
    slice_len : int
        the length of window on the array


    Yelds
    -----
    numpy.ndarray
        Actually yields, no return

    """
    for i in range(len(array_sample) - slice_len + 1):
        yield array_sample[i : i + slice_len]


def slices_jump_slider(
    array_sample: np.ndarray, slice_len: int, jump: int
) -> Generator[np.ndarray, None, None]:
    """Produce sequential slices of an array separated by a jump.

    This function produces continuous sequential slices over an
    array of a certain length spaced out by a 'jump'.
    The function yields, does not return these slices.


    Parameters
    ----------
    array_sample : numpy.ndarray
        array containing the signal
    slice_len : int
        the length of window on the array
    jump : int
        the amount by which the window is moved at iteration


    Yelds
    -----
    numpy.ndarray
        Actually yields, no return

    """
    for start in range(0, len(array_sample) - slice_len + 1, jump):
        yield array_sample[start : start + slice_len]


def ranges_of(array: np.ndarray) -> tuple:
    """Compute the ranges of a binary array.

    This function is made to work with :class:`Range` class objects, such
    that is selects ranges and returns tuples of boundaries.


    Parameters
    ----------
    array : numpy.ndarray
        array


    Returns
    -------
    tuple
        range_return

    """
    marks = np.logical_xor(array[1:], array[:-1])
    boundaries = np.hstack(
        (np.zeros(1), np.where(marks != 0)[0], np.zeros(1) + len(array) - 1)
    )
    if not array[0]:
        boundaries = boundaries[1:]
    if len(boundaries) % 2 != 0:
        boundaries = boundaries[:-1]
    return tuple(
        Range(*boundaries[i : i + 2]) for i in range(0, len(boundaries), 2)
    )


def intersections(left: list, right: list) -> list:
    """Compute the intersection of two lists of ranges.

    This function works over two arrays, `left` and
    `right`, and allows a picking based on intersections. It
    only takes ranges on the left that intersect ranges on the right.


    Parameters
    ----------
    left : List[Range]
        List of ranges
    right : List[Range]
        List of ranges


    Returns
    -------
    result: List[Range]
        result: Ranges from the left that intersect ranges from the right.

    """
    i, j = 0, 0
    result = []
    while i < len(left) and j < len(right):
        lelt, relt = left[i], right[j]
        if lelt.intersects(relt):
            result.append(lelt)
            i += 1
        elif relt.precedes(lelt):
            j += 1
        elif lelt.precedes(relt):
            i += 1
        else:  # relt ends exactly where lelt starts
            j += 1
    return result


def raw_overlap_percent(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """Compute the raw overlap percentage between two binary signals.

    This function takes two binary 0 or 1 signal arrays and gives
    the percentage of overlap.

    Parameters
    ----------
    signal1 : numpy.ndarray
        Binary signal 1
    rsignal2 : numpy.ndarray
        Binary signal 2
    signal2 : np.ndarray


    Returns
    -------
    _raw_overlap_percent: float
        _raw_overlap_percent: Raw overlap percent

    """
    if len(signal1) != len(signal2):
        print("Warning: length of arrays is not matched")
        longer_signal_len = np.max([len(signal1), len(signal2)])
    else:
        longer_signal_len = len(signal1)

    return sum(signal1.astype(int) & signal2.astype(int)) / longer_signal_len


def merge(left: list, right: list) -> list:
    """Merge two lists based linear order.

    Parameters
    ----------
    left : List
        First list
    right : List
        Second list


    Returns
    -------
    type
        output: Merged lists
        :rtpe output: List

    """
    # Initialize an empty list output that will be populated
    # with sorted elements.
    # Initialize two variables i and j which are used pointers when
    # iterating through the lists.
    output = []
    i = j = 0

    # Executes the while loop if both pointers i and j are less than
    # the length of the left and right lists
    while i < len(left) and j < len(right):
        # Compare the elements at every position of both
        # lists during each iteration
        if left[i] < right[j]:
            # output is populated with the lesser value
            output.append(left[i])
            # 10. Move pointer to the right
            i += 1
        else:
            output.append(right[j])
            j += 1
    # The remnant elements are picked from the current
    # pointer value to the end of the respective list
    output.extend(left[i:])
    output.extend(right[j:])

    return output


def scale_arrays(
    array: np.ndarray, maximum: float, minimum: float
) -> np.ndarray:
    """Scale arrays to a maximum and minimum value.

    This function will scale all arrays along the vertical axis to have an
    absolute maximum value of the maximum parameter.


    Parameters
    ----------
    array : numpy.ndarray
        Original signal array with any number iflayers
    maximum : float
        the absolute maximum below which the new array exists
    minimum : float
        the absolute maximum below which the new array exists


    Returns
    -------
    reformed: ~numpy.ndarray
        reformed: a new array with absolute max of maximum

    """
    return np.interp(array, (array.min(), array.max()), (maximum, minimum))


def delay_embedding(
    data: np.ndarray, emb_dim: int, lag: int = 1
) -> np.ndarray:
    """Perform a time-delay embedding of a time series.

    The following code is adapted from openly licensed code written by
    Christopher Schölzel in his package nolds (NOnLinear measures for Dynamical
    Systems).It performs a time-delay embedding of a time series.


    Parameters
    ----------
    data : array
        array-like
    emb_dim : int
        the embedded dimension
    lag : int
        the lag between elements in the embedded vectors

    Returns
    -------
    matrix_vectors: ~nd.array
        matrix_vectors: the embedded vectors

    """
    data = np.asarray(data)
    min_len = (emb_dim - 1) * lag + 1
    if len(data) < min_len:
        msg = (
            "cannot embed data of length {} with embedding dimension {} "
            "and lag {}, minimum required length is {}"
        )
        raise ValueError(msg.format(len(data), emb_dim, lag, min_len))
    m = len(data) - min_len + 1
    indices = np.repeat([np.arange(emb_dim) * lag], m, axis=0)
    indices += np.arange(m).reshape((m, 1))
    return data[indices]


def save_preprocessed(array: np.ndarray, out_fname: str, force: bool) -> None:
    """Save preprocessed arrays to a file.

    This function is written to be called by the cli module.
    It stores arrays in a directory.


    Parameters
    ----------
    array : nd.array
        array to be stored
    out_fname : str
        output file name
    force : bool
        force writing the file


    Returns
    -------
    None
        None

    """
    if not force and Path(out_fname).is_file():
        return
    with contextlib.suppress(FileExistsError):
        Path(out_fname).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_fname, array, allow_pickle=False)


def derivative(
    signal: np.ndarray, fs: int, window_s: int | None = None
) -> np.ndarray:
    """Calculate the first derivative of a signal.

    This function calculates the first derivative of a signal. If
    window_s is given, the signal is smoothed before derivative calculation.


    Parameters
    ----------
    signal : numpy.ndarray
        signal to calculate the derivate over
    fs : int
        sampling rate
    window_s : int
        centralised averaging window length in samples

    Returns
    -------
    dsignal_dt: ~numpy.ndarray
        dsignal_dt: The 1st derivative of the signal length len(signal)-1.

    """
    if window_s is not None:
        # Moving average over signal
        signal_series = pd.Series(signal)
        signal_moving_average = (
            signal_series.rolling(window=window_s, center=True).mean().values
        )
        dsignal_dt = (
            signal_moving_average[1:] - signal_moving_average[:-1]
        ) * fs
    else:
        dsignal_dt = (signal[1:] - signal[:-1]) * fs

    return dsignal_dt


def bell_curve(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Fit a bell curve to the samples of `x`.

    This function calculates a bell curve on the samples of `x`, shifted by
    b, amplified by a for a standard amplitude of 1.


    Parameters
    ----------
    x : numpy.ndarray
        x values to calculate the bell_curve for
    a : float
        amplitude of the bell-curve
    b : float
        time shift of the bell-curve along the x-axis.
    c : float
        steepness factor of bell-curve.


    Returns
    -------
    numpy.ndarray
        bell curve

    """
    return a * np.exp(-((x - b) ** 2) / c**2)


def running_smoother(array: np.ndarray) -> np.ndarray:
    """Apply a running average smoother to an array.

    This is the smoother to use in time calculations.


    Parameters
    ----------
    array : numpy.ndarray
        array to be smoothed


    Returns
    -------
    smoothed_array: ~numpy.ndarray
        smoothed_array: smoothed array

    """
    n_samples = len(array) // 10
    new_list = np.convolve(abs(array), np.ones(n_samples), "valid") / n_samples
    zeros = np.zeros(n_samples - 1)
    return np.hstack((new_list, zeros))


def get_dict_key_where_value(dictionary: dict, value: Any) -> str | None:  # noqa: ANN401
    """Return the key of a dictionary where the value matches the input value.

    This function returns the key of a dictionary where the value matches
    the input value.


    Parameters
    ----------
    dictionary : dictionary: dict
        dictionary to search
    value : any
        value to search for


    Returns
    -------
    key: any
        key: key where value is found

    """
    for key, val in dictionary.items():
        if val == value:
            return key
    return None
