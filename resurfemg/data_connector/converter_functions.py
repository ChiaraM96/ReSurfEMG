"""Standardized EMG file type converters.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains standardized functions to work with various EMG file types
from various hardware/software combinations, and convert them down to
an array that can be further processed with other modules.
"""

from __future__ import annotations

import logging
import platform
import re
import warnings
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pandas as pd
import scipy.io as sio

from resurfemg.data_connector.tmsisdk_lite import Poly5Reader

if platform.system() == "Windows":
    from resurfemg.data_connector.adicht_reader import AdichtReader


logger = logging.getLogger(__name__)


def load_poly5(file_path: str, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load a .Poly5 file and return the data as a pandas DataFrame.

    This function loads a .Poly5 file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the sampling rate,
    loaded channels, and units.

    Args:
        file_path (str): Path to the file to be loaded.
        verbose (bool): Print verbose output.

    Returns:
        tuple:
            - pandas.DataFrame: Pandas DataFrame of the loaded data.
            - dict: Metadata of the loaded data.
    """
    if verbose:
        logger.info("Loading .Poly5 ...")
    poly5_data = Poly5Reader(file_path, verbose=verbose)
    if verbose:
        logger.info("Loaded .Poly5, extracting data ...")
    n_samples = poly5_data.num_samples
    loaded_data = poly5_data.samples[:, :n_samples]
    metadata = {}
    metadata["fs"] = poly5_data.sample_rate
    metadata["labels"] = list(poly5_data.ch_names)
    metadata["units"] = list(poly5_data.ch_unit_names)
    data_df = pd.DataFrame(np.asarray(loaded_data.T), columns=metadata["labels"])
    if verbose:
        logger.info("Loading data completed")

    return data_df, metadata


def load_mat(
    file_path: str, key_name: str | None = None, verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Load a .mat file and return the data as a pandas DataFrame.

    This function loads a .mat file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the sampling rate,
    loaded channels, and units.

    Args:
        file_path (str): Path to the file to be loaded.
        key_name (str): Key name for .mat files.
        verbose (bool): Print verbose output.

    Returns:
        tuple:
            - pandas.DataFrame: Pandas DataFrame of the loaded data.
            - dict: Metadata of the loaded data.

    Raises:
        TypeError: If no key_name is provided.
    """
    if verbose:
        logger.info("Loading .mat ...")
    mat_dict = sio.loadmat(file_path, mdict=None, appendmat=False)
    if verbose:
        logger.info("Loaded .mat, extracting data ...")
    if isinstance(key_name, str):
        loaded_data = mat_dict[key_name]
        if loaded_data.shape[0] > loaded_data.shape[1]:
            loaded_data = np.rot90(loaded_data)
            if verbose:
                logger.info("Transposed loaded data.")
        data_df = pd.DataFrame(loaded_data.T)
        logger.info("Loading data completed")
    else:
        msg = "No key_name provided."
        raise TypeError(msg)

    return data_df, {}


def load_csv(
    file_path: str, force_col_reading: bool, verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Load a .csv file and return the data as a pandas DataFrame.

    This function loads a .csv file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the loaded channels.

    Args:
        file_path (str): Path to the file to be loaded.
        force_col_reading (bool): Force column reading for row based .csv
            files.
        verbose (bool): Print verbose output.

    Returns:
        tuple:
            - pandas.DataFrame: Pandas DataFrame of the loaded data.
            - dict: Metadata of the loaded data.

    Raises:
        UserWarning: If the .csv is row based and force_col_reading is not
            True.
    """

    def has_header(file_path: str, nrows: int = 20) -> bool:
        df = pd.read_csv(file_path, header=None, nrows=nrows)
        df_header = pd.read_csv(filepath_or_buffer=file_path, nrows=nrows)
        return tuple(df.dtypes) != tuple(df_header.dtypes)

    def chech_row_wise(file_path: str, nrows: int = 20) -> bool:
        with Path(file_path).open("r") as f:
            n_lines = sum(1 for _ in f)

        with Path(file_path).open("r") as f:
            col_lg_row = 0
            i = 0
            for i, line in enumerate(f):
                if len(line) > n_lines:
                    col_lg_row += 1
                if i > nrows or col_lg_row > nrows:
                    break
            return not (col_lg_row > nrows // 2 or col_lg_row == n_lines)

    if verbose:
        logger.info("Loading .csv ...")
    row_wise = chech_row_wise(file_path, nrows=20)
    if (row_wise is False) and (force_col_reading is not True):
        msg = [
            "The provided .csv is row based. ",
            "This could yield significant loading durations.",
            "If you want to proceed, set force_col_reading=True",
        ]
        raise UserWarning(msg)

    metadata = {}
    if verbose:
        logger.info("Loaded .csv, extracting data ...")
    if row_wise and has_header(file_path):
        data_df = pd.read_csv(file_path)
        metadata["labels"] = data_df.columns.values
    else:
        csv_data = pd.read_csv(file_path, header=None)
        data_df = pd.DataFrame(csv_data.to_numpy())
    logger.info("Loading data completed")

    return data_df, metadata


def load_npy(file_path: str, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """This function loads a .npy file and returns the data as a numpy array.

    Args:
        file_path (str): Path to the file to be loaded.
        verbose (bool): Print verbose output.

    Returns:
        tuple:
            - pandas.DataFrame: Pandas DataFrame of the loaded data.
            - dict: Metadata of the loaded data.
    """
    logger.info("Loaded .npy, extracting data ...")
    np_data = np.load(file_path)
    if np_data.shape[0] > np_data.shape[1]:
        np_data = np.rot90(np_data)
        if verbose:
            logger.info("Transposed loaded data.")
    data_df = pd.DataFrame(np_data)
    metadata = {}

    return data_df, metadata


def load_adicht(
    file_path: str,
    record_idx: int,
    channel_idxs: list[int] | None = None,
    resample_channels: dict[int, int] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load a .adicht file and return the data as a pandas DataFrame.

    This function loads a .adicht file and returns the data as a pandas
    DataFrame.

    Args:
        file_path (str): Path to the file to be loaded.
        record_idx (int): The record index to extract data from.
        channel_idxs (list[int], optional): List of channel indices to
            extract.
        resample_channels (dict[int, int], optional): Map of channel_idx to
            target sampling rate.
        verbose (bool): Print verbose output.

    Returns:
        tuple:
            - pandas.DataFrame: Pandas DataFrame of the loaded data.
            - dict: Metadata of the loaded data.

    Raises:
        OSError: If not run on Windows.
    """
    if platform.system() != "Windows":
        msg = "AdichtReader only availabe on Windows."
        raise OSError(msg)

    if verbose:
        logger.info("Loading .adicht ...")

    # Extract the data
    if verbose:
        logger.info("Loaded .adicht, extracting data ...")
    adicht_data = AdichtReader(  # pyright: ignore[reportPossiblyUnboundVariable]
        file_path
    )
    if verbose:
        adicht_data.print_metadata()
    adi_meta = adicht_data.generate_metadata()
    if channel_idxs is None:
        channel_idxs = list(adicht_data.channel_map.keys())
    fs_sel = [adi_meta[idx]["fs"][0] for idx in channel_idxs]
    if len(set(fs_sel)) > 1:
        warnings.warn("""\nMultiple sampling rates detected, which cannot be parsed into
            one numpy array. Please specify the channel_idxs to select a
            subset of channels with the same sampling rate or resample
            channels with the resample_channels argument.\n""")
        max_fs = max(fs_sel)
        channel_idxs = [i for i, fs in enumerate(fs_sel) if fs == max_fs]

    data_df, fs_emg = adicht_data.extract_data(
        channel_idxs=channel_idxs,
        record_idx=record_idx,
        resample_channels=resample_channels,
    )
    metadata = {
        "fs": fs_emg,
        "channel_idxs": channel_idxs,
        "labels": adicht_data.get_labels(channel_idxs),
        "units": adicht_data.get_units(channel_idxs, record_idx),
        "record_id": record_idx,
    }
    if verbose:
        logger.info("Loading data completed")

    return data_df, metadata


def _load_by_extension(
    file_path: str, file_ext: str, file_extension: str, verbose: bool, kwargs: dict
) -> tuple[pd.DataFrame, dict]:
    loaders = {
        "poly5": load_poly5,
        "mat": lambda fp, v: load_mat(fp, kwargs.get("key_name", ""), v),
        "csv": lambda fp, v: load_csv(fp, kwargs.get("force_col_reading", False), v),
        "npy": load_npy,
    }
    if file_ext in loaders:
        return loaders[file_ext](file_path, verbose)
    if file_ext.startswith("adi"):
        return load_adicht(
            file_path,
            record_idx=kwargs.get("record_idx", 0),
            channel_idxs=kwargs.get("channel_idxs"),
            resample_channels=kwargs.get("resample_channels"),
            verbose=verbose,
        )
    msg = f"No methods available for file extension {file_extension}."
    raise UserWarning(msg)


def _rename_channels(
    data_df: pd.DataFrame, kwargs: dict, verbose: bool
) -> pd.DataFrame:
    labels = kwargs.get("labels")
    if isinstance(labels, list) and len(labels) == data_df.shape[1]:
        if not all(isinstance(channel, str) for channel in labels):
            msg = "All channel names should be str"
            raise TypeError(msg)
        if len(labels) != len(set(labels)):
            msg = "Channel names should be unique"
            raise UserWarning(msg)
        if verbose:
            logger.info(
                "Renamed channels: %s", list(zip(data_df.columns, labels, strict=False))
            )
        data_df.columns = labels
    return data_df


def _select_channels(
    data_df: pd.DataFrame, metadata: dict, kwargs: dict, verbose: bool
) -> pd.DataFrame:
    channel_idxs = kwargs.get("channel_idxs", list(range(data_df.shape[1])))
    if not all(
        isinstance(idx, int) and 0 <= idx < data_df.shape[1] for idx in channel_idxs
    ):
        msg = "channel_idxs should be a list of ints"
        raise TypeError(msg)
    data_df = cast("pd.DataFrame", data_df.iloc[:, channel_idxs])
    metadata["channel_idxs"] = channel_idxs
    for item in ["labels", "units"]:
        if item in metadata:
            metadata[item] = [metadata[item][idx] for idx in channel_idxs]
    if verbose:
        logger.info("Selected channels: %s", channel_idxs)
    return data_df


def load_file(
    file_path: str, verbose: bool = True, **kwargs: dict
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Load a file as numpy array.

    This function loads a file from a given path and returns the data as a
    numpy array. The function can handle .poly5, .mat, .csv, and .npy files.
    The function can also rename channels and drop channels from the data.

    Args:
        file_path (str): Path to the file to be loaded.
        verbose (bool): Print verbose output.
        **kwargs: Additional keyword arguments for specific file loaders:

            - key_name (str): Key name for loading .mat files.
            - force_col_reading (bool): If True, force reading columns for
              .csv files. Default is False.
            - record_idx (int): Record index for loading .adi* files.
              Default is 0.
            - channel_idxs (list): List of channel indices for loading .adi
              files.
            - labels (list): List of new channel names to rename the columns.

    Returns:
        tuple:
            - numpy.ndarray: Numpy array of the loaded data.
            - pandas.DataFrame: Pandas DataFrame of the loaded data.
            - dict: Metadata of the loaded data.

    Raises:
        TypeError: If file_path is not a str.
    """
    if not isinstance(file_path, str):
        msg = "file_path should be a str."
        raise TypeError(msg)
    file_name = Path(file_path).name
    file_extension = file_name.split(".")[-1]
    file_ext = file_extension.lower()
    if verbose:
        logger.info("Detected .%s", file_ext)

    data_df, metadata = _load_by_extension(
        file_path, file_ext, file_extension, verbose, kwargs
    )
    metadata["file_name"] = file_name
    metadata["file_dir"] = Path(file_path).parent
    metadata["file_extension"] = file_extension

    data_df = _rename_channels(data_df, kwargs, verbose)
    if not file_ext.startswith("adi"):
        data_df = _select_channels(data_df, metadata, kwargs, verbose)

    np_data = np.flipud(np.rot90(data_df.to_numpy(), axes=(0, 1)))
    for item in ["fs", "labels", "units"]:
        if item not in metadata:
            logger.info("Metadata %s not found. Set it manually.", item)
    return np_data, data_df, metadata


def poly5unpad(to_be_read: str) -> np.ndarray:
    """Converts a Poly5 read into an array without padding.

    This padding is a
    quirk in the python Poly5 interface that pads with zeros on the end.

    Args:
        to_be_read (str): Filename of python read Poly5.

    Returns:
        numpy.ndarray: Unpadded array.
    """
    read_object = Poly5Reader(to_be_read)
    sample_number = read_object.num_samples
    return read_object.samples[:, :sample_number]


def matlab5_jkmn_to_array(file_name: str) -> np.ndarray:
    """LEGACY FUNCTION.

    This file reads matlab5 files as produced in the Jonkman laboratory, on the
    Biopac system and returns arrays in the format and shape our the ReSurfEMG
    functions work on.

    Args:
        file_name (str): Filename of matlab5 files.

    Returns:
        numpy.ndarray: Arrayed data.
    """
    file = sio.loadmat(file_name, mdict=None, appendmat=False)
    arrayed = np.rot90(file["data_emg"])
    output_copy = arrayed.copy()
    arrayed[4] = output_copy[0]
    arrayed[3] = output_copy[1]
    arrayed[1] = output_copy[3]
    arrayed[0] = output_copy[4]
    return arrayed


def csv_from_jkmn_to_array(file_name: str) -> np.ndarray:
    """LEGACY FUNCTION.

    This function takes a file from the Jonkman lab in csv format and changes
    it into the shape the library functions work on.

    Args:
        file_name (str): Filename of csv files.

    Returns:
        numpy.ndarray: Arrayed data.
    """
    file = pd.read_csv(file_name)
    new_df = (
        file.T.reset_index()
        .T.reset_index(drop=True)
        .set_axis([f"lead.{i + 1}" for i in range(file.shape[1])], axis=1)
    )
    arrayed = np.rot90(new_df)
    return np.flipud(arrayed)


def poly_dvrman(file_name: str) -> np.ndarray:
    """LEGACY FUNCTION.

    This is a function to read in Duiverman type Poly5 files, which has 18
    layers/pseudo-leads, and return an array of the twelve  unprocessed leads
    for further pre-processing. The leads eliminated were RMS calculated on
    other leads (leads 6-12). The expected organization returned is from leads
    0-5 EMG data, then the following leads
    # 6 Paw: airway pressure (not always recorded)
    # 7 Pes: esophageal pressure (not always recorded)
    # 8 Pga: gastric pressure (not always recorded)
    # 9 RR: respiratory rate I guess (very unreliable)
    # 10 HR: heart rate
    # 11 Tach: number of breath (not reliable)

    Args:
        file_name (str): Filename of Poly5 Duiverman type file.

    Returns:
        numpy.ndarray: Arrayed data.
    """
    data_samples = Poly5Reader(file_name)
    return np.vstack([data_samples.samples[:6], data_samples.samples[12:]])


def dvrmn_csv_to_array(file_name: str) -> np.ndarray:
    """LEGACY FUNCTION.

    Transform an already preprocessed csv from the Duiverman lab into an EMG
    in the format our other functions can work on it. Note that some
    preprocessing steps are already applied so pipelines may need adjusting.

    Args:
        file_name (str): Filename of csv file.

    Returns:
        numpy.ndarray: Arrayed data.
    """
    file = pd.read_csv(file_name)
    new_df = file.drop(["Events", "Time"], axis=1)
    arrayed = np.rot90(new_df)
    return np.flipud(arrayed)


def dvrmn_csv_freq_find(file_name: str) -> int:
    """LEGACY FUNCTION.

    Extract the sampling rate of a Duiverman type csv of EMG. Note
    this data may be resampled down by a factor of 10.

    Args:
        file_name (str): Filename of csv file.

    Returns:
        int: Sampling frequency.
    """
    file = pd.read_csv(file_name)
    sample_points = len(file)
    time_string = file["Time"][sample_points - 1]
    seconds = float(time_string[5:10])
    minutes = float(time_string[2:4])
    hours = int(time_string[0:1])
    sum_time = (hours * 3600) + (minutes * 60) + seconds

    return round(sample_points / sum_time)


class BiopacMetadata(TypedDict):
    """Dictionary to store metadata from Biopac CSV files."""

    recording_date: str | None
    sample_time: float | None
    sample_time_unit: str | None
    channel_number: int
    labels: list[str]
    units: list[str]
    exp_name: str | None
    fs: int | None


def load_biopac_csv(
    file: str, amplifier_model: str = "EMG100C"
) -> tuple[pd.DataFrame, BiopacMetadata]:
    """Import data from a Biopac CSV file and return the data as a pandas DataFrame.

    Args:
        file (str): Path to the Biopac CSV file.
        amplifier_model (str): The model of the Biopac amplifier used for the recording.
            Default is "EMG100C". Used in the regex pattern to extract channel names,
            as BIOPAC's exporter automatically appends the amplifier's model
            to the channel names.

    Returns:
        tuple: A `(data_df, metadata)` pair. `data_df` is a pandas.DataFrame with a
            `Time [s]` column (sample timestamps in seconds) plus one column per
            channel, named by label. `metadata` is a dict with the keys:
            `recording_date`: Recording date and time.
            `sample_time`: Sample timestamp from the start of the recording, in seconds.
            `sample_time_unit`: Original unit of the sample time.
            `channel_number`: Number of channels in the recording.
            `labels`: Labels of the channels.
            `units`: Units of the channels.
            `exp_name`: Experiment name.
    """
    scalar_tokens = {
        "recording_date": re.compile(
            r"Recording on: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"
        ),
        "sample_time": re.compile(r"([0-9]*\.[0-9]*) [m]?sec"),
        "sample_time_unit": re.compile(r"[0-9]*\.[0-9]* ([m]?sec)"),
        "channel_number": re.compile(r"([0-9]*) channels"),
    }

    channel_name_pattern = re.compile(
        r"^([A-Za-z0-9]+) - " + re.escape(amplifier_model) + r"$"
    )

    metadata: BiopacMetadata = {
        "recording_date": None,
        "sample_time": None,
        "sample_time_unit": None,
        "channel_number": -1,
        "labels": [],
        "units": [],
        "exp_name": None,
        "fs": None,
    }

    row_counter = 0
    with Path(file).open("r") as f:
        total_row_count = sum(1 for row in f)
        f.seek(0)
        _line = f.readline()
        row_counter += 1
        metadata["exp_name"] = _line.strip()
        while (not _line.startswith("sec")) and row_counter < total_row_count:
            _line = f.readline()
            row_counter += 1
            if not len(_line):
                continue
            for key, pattern in scalar_tokens.items():
                match = re.search(pattern, _line)
                if match is not None:
                    metadata[key] = match.group(1)
                    if (
                        key == "channel_number"
                        and metadata["channel_number"] is not None
                    ):
                        metadata["channel_number"] = int(metadata["channel_number"])
                        for _ in range(metadata["channel_number"]):
                            _line = f.readline()
                            row_counter += 1
                            match = re.search(channel_name_pattern, _line)
                            if match is not None:
                                metadata["labels"].append(match.group(1))
                                _line = f.readline().strip()
                                row_counter += 1
                                metadata["units"].append(_line)
                            if _line.startswith("sec"):
                                break
                    continue

    if metadata["sample_time"] is not None:
        scaling = 1e-3 if metadata["sample_time_unit"] == "msec" else 1.0
        metadata["sample_time"] = scaling * float(metadata["sample_time"])
        metadata["fs"] = int(1 / metadata["sample_time"])

    data_df = pd.read_csv(
        filepath_or_buffer=file,
        header=row_counter,
        names=["Time [s]", *metadata["labels"]],
        index_col=None,
        usecols=range(len(metadata["labels"]) + 1),
    )

    return data_df, metadata
