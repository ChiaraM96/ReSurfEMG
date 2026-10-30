"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains standardized functions to work with various EMG file types
from various hardware/software combinations, and convert them down to
an array that can be further processed with other modules.
"""  # noqa: D205

import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from resurfemg.data_connector.tmsisdk_lite import Poly5Reader

if platform.system() == "Windows":
    from resurfemg.data_connector.adicht_reader import AdichtReader

PATH_LIKE = str | Path


def _load_by_extension(
    file_path: PATH_LIKE, verbose: bool, **kwargs
) -> tuple[pd.DataFrame, dict, str]:
    file_path = Path(file_path)
    file_name = file_path.name
    file_dir = file_path.parent
    file_extension = file_name.split(".")[-1]
    file_ext = file_extension.lower()
    if file_extension.lower().startswith("adi"):
        file_ext = "adi"
    loaders = {
        "poly5": load_poly5,
        "mat": lambda fp, v: load_mat(fp, kwargs.get("key_name", ""), v),
        "csv": lambda fp, v: load_csv(fp, kwargs.get("force_col_reading", False), v),
        "npy": load_npy,
        "adi": lambda fp, v: load_adicht(
            fp,
            record_idx=kwargs.get("record_idx", 0),
            channel_idxs=kwargs.get("channel_idxs"),
            resample_channels=kwargs.get("resample_channels"),
            verbose=v,
        ),
    }

    if verbose:
        print(f"Detected .{file_ext}")
    if file_ext in loaders:
        data_df, metadata = loaders[file_ext](file_path, verbose)
    else:
        msg = f"No methods available for file extension {file_extension}."
        raise UserWarning(msg)

    metadata["file_name"] = file_name
    metadata["file_dir"] = file_dir
    metadata["file_extension"] = file_extension

    return data_df, metadata, file_ext


def _validate_channel_labels(
    data_df: pd.DataFrame, labels: list | None, verbose: bool = True
) -> pd.DataFrame:

    if isinstance(labels, list) and len(labels) == data_df.shape[1]:
        if not all(isinstance(channel, str) for channel in labels):
            msg = "All channel names should be str"
            raise TypeError(msg)
        if len(labels) != len(set(labels)):
            msg = "Channel names should be unique"
            raise UserWarning(msg)
        if verbose:
            print(
                "Renamed channels:",
                *zip(data_df.columns, labels, strict=False),
                "...",
            )
        data_df.columns = labels
    return data_df


def load_file(
    file_path: PATH_LIKE, verbose: bool = True, **kwargs
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Load data from a given file path as a numpy array.

    This function loads a file from a given path and returns the data as a
    numpy array. The function can handle .poly5, .mat, .csv, and .npy files.
    The function can also rename channels and drop channels from the data.

    Args:
      file_path(Path | str): Path to the file to be loaded.
      verbose(bool): If True, print verbose output. Defaults to True.
      kwargs: Additional keyword arguments for specific file loaders:
        - key_name (str): Key name for loading .mat files.
        - force_col_reading (bool): If True, force reading columns for .csv files.
            Default is False.
        - record_idx (int): Record index for loading .adi* files. Default is 0.
        - channel_idxs (list): List of channel indices for loading .adi files.
        - labels (list): List of new channel names to rename the columns.

    Returns:
      np.ndarray: Numpy array of the loaded data.
      pd.DataFrame: Pandas DataFrame of the loaded data.
      dict: Metadata of the loaded data.

    """

    def _select_specific_channels(
        data_df: pd.DataFrame, metadata: dict, verbose: bool
    ) -> None:
        channel_idxs = kwargs.get("channel_idxs", list(range(data_df.shape[1])))
        if not all(
            isinstance(idx, int) and 0 <= idx < data_df.shape[1] for idx in channel_idxs
        ):
            msg = "channel_idxs should be a list of ints"
            raise TypeError(msg)
        data_df = data_df.iloc[:, channel_idxs]
        metadata["channel_idxs"] = channel_idxs
        for item in ["labels", "units"]:
            if item in metadata:
                metadata[item] = [metadata[item][idx] for idx in channel_idxs]
        if verbose:
            print("Selected channels:", channel_idxs)

    if not (isinstance(file_path, PATH_LIKE)):
        msg = "file_path should be a Path or a str."
        raise TypeError(msg)

    file_path = Path(file_path)

    # 1. Load File types: .poly5, .mat, .csv,

    data_df, metadata, file_ext = _load_by_extension(file_path, verbose, **kwargs)
    labels = kwargs.get("labels")
    data_df = _validate_channel_labels(data_df, labels, verbose)
    # 2. Select specific channels
    if not file_ext.startswith("adi"):
        _select_specific_channels(data_df, metadata, verbose)

    # 3. Convert remaining float channels to numpy array
    np_data = np.flipud(np.rot90(data_df.to_numpy(), axes=(0, 1)))

    for item in ["fs", "labels", "units"]:
        if item not in metadata:
            print(f"Warning: Metadata {item} not found. Set it manually.")

    return np_data, data_df, metadata


def load_poly5(file_path: PATH_LIKE, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load a .Poly5 file as a pandas DataFrame and extract metadata.

    This function loads a .Poly5 file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the sampling rate,
    loaded channels, and units.

    Args:
      file_path(Path | str): Path to the .Poly5 file to be loaded.
      verbose(bool): If True, print verbose output. Defaults to True.

    Returns:
      pd.DataFrame: Pandas DataFrame of the loaded data.
      dict: Metadata of the loaded data.

    """
    if verbose:
        print("Loading .Poly5 ...")
    poly5_data = Poly5Reader(file_path, verbose=verbose)
    if verbose:
        print("Loaded .Poly5, extracting data ...")
    n_samples = poly5_data.num_samples
    loaded_data = poly5_data.samples[:, :n_samples]
    metadata = {
        "fs": poly5_data.sample_rate,
        "labels": poly5_data.ch_names,
        "units": poly5_data.ch_unit_names,
    }
    data_df = pd.DataFrame(loaded_data.T, columns=pd.Index(metadata["labels"]))
    if verbose:
        print("Loading data completed")

    return data_df, metadata


def load_mat(
    file_path: PATH_LIKE, key_name: str, verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Load a .mat file as a pandas DataFrame and extract metadata.

    This function loads a .mat file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the sampling rate,
    loaded channels, and units.

    Args:
        file_path (Path | str): Path to the file to be loaded
        key_name (str): Key name for .mat files
        verbose (bool, optional): If True, print verbose output. Defaults to True.

    Returns:
        tuple[~pandas.DataFrame, dict]: A tuple containing the loaded data as
            a pandas DataFrame and the extracted metadata.

    """
    if verbose:
        print("Loading .mat ...")
    mat_dict = sio.loadmat(file_path, mdict=None, appendmat=False)
    if verbose:
        print("Loaded .mat, extracting data ...")
    if isinstance(key_name, str):
        loaded_data = mat_dict[key_name]
        if loaded_data.shape[0] > loaded_data.shape[1]:
            loaded_data = np.rot90(loaded_data)
            if verbose:
                print("Transposed loaded data.")
        data_df = pd.DataFrame(loaded_data.T)
        print("Loading data completed")
    else:
        msg = "No key_name provided."
        raise TypeError(msg)

    return data_df, {}


def _has_header(file_path: PATH_LIKE, nrows: int = 20) -> bool:
    df = pd.read_csv(file_path, header=None, nrows=nrows)
    df_header = pd.read_csv(file_path, nrows=nrows)
    return tuple(df.dtypes) != tuple(df_header.dtypes)


def _chech_row_wise(file_path: PATH_LIKE, nrows: int = 20) -> bool:
    file_path = Path(file_path)
    with file_path.open("r") as f:
        n_lines = sum(1 for _ in f)

    with file_path.open("r") as f:
        col_lg_row = 0
        for i, line in enumerate(f):
            if len(line) > n_lines:
                col_lg_row += 1
            if i > nrows or col_lg_row > nrows:
                break
        return not (col_lg_row > nrows // 2 or col_lg_row == n_lines)


def load_csv(
    file_path: PATH_LIKE, force_col_reading: bool, verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Load a .csv file as a pandas DataFrame and extract metadata.

    This function loads a .csv file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the loaded channels.

    Args:
      file_path(str): Path to the file to be loaded
      force_col_reading(bool): Force column reading for row based .csv files
      verbose(bool, optional): Print verbose output (Default value = True)

    Returns:
      tuple[~pandas.DataFrame, dict]: A tuple containing the loaded data as
            a pandas DataFrame and the extracted metadata.
    """
    file_path = Path(file_path)
    if verbose:
        print("Loading .csv ...")
    row_wise = _chech_row_wise(file_path, nrows=20)
    if (row_wise is False) and (force_col_reading is not True):
        msg = (
            "The provided .csv is row based. This could yield "
            "significant loading durations. If you want to "
            "proceed, set force_col_reading=True"
        )
        raise UserWarning(msg)

    metadata = {}
    if row_wise and _has_header(file_path):
        data_df = pd.read_csv(file_path)
        if verbose:
            print("Loaded .csv, extracting data ...")
        metadata["labels"] = data_df.columns.values
    else:
        if verbose:
            print("Loaded .csv, extracting data ...")
        csv_data = pd.read_csv(file_path, header=None)
        data_df = pd.DataFrame(csv_data.to_numpy())
    print("Loading data completed")

    return data_df, metadata


def load_npy(file_path: PATH_LIKE, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """This function loads a .npy file and returns the data as a numpy array.

    Args:
      file_path(str): Path to the file to be loaded
      verbose(bool, optional): Print verbose output (Default value = True)

    Returns:
      tuple[~pandas.DataFrame, dict]: A tuple containing the loaded data as
            a pandas DataFrame and the extracted metadata.

    """
    print("Loaded .npy, extracting data ...")
    np_data = np.load(file_path)
    if np_data.shape[0] > np_data.shape[1]:
        np_data = np.rot90(np_data)
        if verbose:
            print("Transposed loaded data.")
    data_df = pd.DataFrame(np_data)
    metadata = {}

    return data_df, metadata


def load_adicht(
    file_path: PATH_LIKE,
    record_idx: int,
    channel_idxs: list[int] | None = None,
    resample_channels: dict[int, int] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """This function loads a .adicht file and returns the data as a numpy array.

    Args:
      file_path(Path | str): Path to the file to be loaded
      verbose(bool, optional): Print verbose output (Default value = True)
      record_idx(int): Index of the record to be loaded
      channel_idxs (list[int] | None, optional): List of channel indices to be
        loaded (Default value = None)
      resample_channels (dict[int, int] | None, optional): Dictionary mapping
        channel indices to their new sampling rates (Default value = None)

    Returns:
      tuple[~pandas.DataFrame, dict]: A tuple containing the loaded data as
            a pandas DataFrame and the extracted metadata.
    """
    if platform.system() != "Windows":
        msg = "AdichtReader only availabe on Windows."
        raise UserWarning(msg)
    if verbose:
        print("Loading .adicht ...")

    # Extract the data
    if verbose:
        print("Loaded .adicht, extracting data ...")
    adicht_data = AdichtReader(str(file_path))
    if verbose:
        adicht_data.print_metadata()
    adi_meta = adicht_data.generate_metadata()
    if channel_idxs is None:
        channel_idxs = list(adicht_data.channel_map.keys())
    fs_sel = [adi_meta[idx]["fs"][0] for idx in channel_idxs]
    if len(set(fs_sel)) > 1:
        warnings.warn(
            """\nMultiple sampling rates detected, which cannot be parsed into
            one numpy array. Please specify the channel_idxs to select a
            subset of channels with the same sampling rate or resample
            channels with the resample_channels argument.\n"""
        )
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
        print("Loading data completed")

    return data_df, metadata


def poly5unpad(to_be_read: PATH_LIKE) -> np.ndarray:
    """Converts a Poly5 read into an array without padding.

    This padding is a quirk in the python Poly5 interface that pads with
    zeros on the end.

    Args:
      to_be_read(Path | str): Filename of python read Poly5

    Returns:
        ~numpy.ndarray: Unpadded array

    """
    read_object = Poly5Reader(str(to_be_read))
    sample_number = read_object.num_samples
    return read_object.samples[:, :sample_number]


def matlab5_jkmn_to_array(file_name: PATH_LIKE) -> np.ndarray:
    """LEGACY FUNCTION.

    This file reads matlab5 files as produced in the Jonkman laboratory, on the
    Biopac system and returns arrays in the format and shape our the ReSurfEMG
    functions work on.

    Args:
      file_name(Path | str): Filename of matlab5 files

    Returns:
      ~numpy.ndarray: Arrayed data

    """
    file = sio.loadmat(str(file_name), mdict=None, appendmat=False)
    arrayed = np.rot90(file["data_emg"])
    output_copy = arrayed.copy()
    arrayed[4] = output_copy[0]
    arrayed[3] = output_copy[1]
    arrayed[1] = output_copy[3]
    arrayed[0] = output_copy[4]
    return arrayed


def csv_from_jkmn_to_array(file_name: PATH_LIKE) -> np.ndarray:
    """LEGACY FUNCTION.

    This file reads csv files as produced in the Jonkman laboratory, on the
    Biopac system and returns arrays in the format and shape our the ReSurfEMG
    functions work on.

    Args:
      file_name(Path | str): Filename of csv files

    Returns:
      ~numpy.ndarray: Arrayed data

    """
    file = pd.read_csv(file_name)
    new_df = (
        file.T.reset_index()
        .T.reset_index(drop=True)
        .set_axis([f"lead.{i + 1}" for i in range(file.shape[1])], axis=1)
    )
    arrayed = np.rot90(new_df)
    return np.flipud(arrayed)


def poly_dvrman(file_name: PATH_LIKE) -> np.ndarray:
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
    # 11 Tach: number of breath (not reliable).

    Args:
      file_name(Path | str): Filename of Poly5 Duiverman type file

    Returns:
      ~numpy.ndarray: Arrayed data

    """
    data_samples = Poly5Reader(file_name)
    return np.vstack([data_samples.samples[:6], data_samples.samples[12:]])


def dvrmn_csv_to_array(file_name: PATH_LIKE) -> np.ndarray:
    """LEGACY FUNCTION.

    Transform an already preprocessed csv from the Duiverman lab into an EMG
    in the format our other functions can work on it. Note that some
    preprocessing steps are already applied so pipelines may need adjusting.

    Args:
      file_name(str): Filename of csv file

    Returns:
      numpy.ndarray: arrayed data

    """
    file = pd.read_csv(file_name)
    new_df = file.drop(["Events", "Time"], axis=1)
    arrayed = np.rot90(new_df)
    return np.flipud(arrayed)


def dvrmn_csv_freq_find(file_name: PATH_LIKE) -> int:
    """LEGACY FUNCTION.

    Extract the sampling rate of a Duiverman type csv of EMG. Note
    this data may be resampled down by a factor of 10.

    Args:
      file_name(Path | str): Filename of csv file

    Returns:
      int: Sampling frequency

    """
    file = pd.read_csv(file_name)
    sample_points = len(file)
    time_string = file["Time"][sample_points - 1]
    seconds = float(time_string[5:10])
    minutes = float(time_string[2:4])
    hours = int(time_string[0:1])
    sum_time = (hours * 3600) + (minutes * 60) + seconds
    return round(sample_points / sum_time)
