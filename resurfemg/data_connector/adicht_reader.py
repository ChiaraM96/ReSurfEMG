"""Class for loading timeseries data from an ADInstruments device.

The class AdichtReader is designed to load EMG data from an ADInstruments
device using the .adicht file format (Labchart) and prepares it for use in
ReSurfEMG. The foundation of the AdichtReader class is the repository
'adinstruments_sdk_python' by Jim Hokanson, available at:
https://github.com/JimHokanson/adinstruments_sdk_python.

An example of how to use this class is provided in the main block of this file.
This example executes only if the script is run directly by the Python
interpreter and not when imported as a module.
"""

import platform
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from resurfemg.helper_functions.math_operations import get_dict_key_where_value

if TYPE_CHECKING:
    from adi.read import File as AdiFile

if platform.system() == "Windows":
    import adi


class AdichtReader:
    """Class for loading timeseries data from an ADInstruments.

    Class for loading timeseries data from an ADInstruments devices using
    the .adicht/.adidat/.adibin file formats (LabChart, BIOPAC) and prepare it
    for use in ReSurfEMG.
    Based on the 'adinstruments_sdk_python' repository by Jim Hokanson,
    available at: https://github.com/JimHokanson/adinstruments_sdk_python.
    """

    def __init__(self, file_path: str):
        """:param file_path: The file path to the import"""
        if platform.system() != "Windows":
            msg = "AdichtReader is only available on Windows."
            raise ImportError(msg)
        self.file_path = file_path
        self.metadata: list[dict]  # List of metadata dicts per channel
        self.metadata_table: PrettyTable  # PrettyTable object for displaying metadata
        self.channel_map: dict[int, int]  # Dictionary mapping channel names to IDs
        self.adicht_data: AdiFile  # Reader object for the file
        self.record_map: dict[int, int]  # Dictionary mapping record idx to IDs

        self._validate_file_path()
        self._initialize_reader()
        self._initialize_channel_map()
        self._initialize_record_map()

    def _validate_file_path(self) -> None:
        """Validates whether the provided file path exists and is readable."""
        if not Path(self.file_path).exists():
            msg = f"The file '{self.file_path}' was not found."
            raise FileNotFoundError(msg)
        if not Path(self.file_path).is_file():
            msg = f"The path '{self.file_path}' does not refer to a file."
            raise ValueError(msg)

    def _initialize_reader(self) -> None:
        """Initializes the adi-reader and loads the file."""
        try:
            self.adicht_data = adi.read_file(self.file_path)
        except Exception as e:
            msg = f"Error loading the file: {e}"
            raise RuntimeError(msg) from e

    def _initialize_channel_map(self) -> None:
        """Creates a dictionary mapping the channel names to their IDs."""
        self.channel_map = {
            i: channel.id for i, channel in enumerate(self.adicht_data.channels)
        }

    def _initialize_record_map(self) -> None:
        """Creates a dictionary mapping the channel names to their idxs."""
        self.record_map = {
            i: record.id for i, record in enumerate(self.adicht_data.records)
        }

    def __repr__(self):
        return f"<AdichtReader(file_path={self.file_path})>"

    def generate_metadata(self) -> list[dict]:
        """Generate metadata for each channel in the ADInstruments file.

        Extracts metadata on channels, samples, records, sampling rates, units,
        and time step and sets it in self.metadata and self.metadata_table.

        :return: List of metadata dicts per channel
        :rtype: list[dict]
        """
        table = PrettyTable()
        table.field_names = [
            "idx",
            "Channel ID",
            "Name",
            "Records",
            "Samples",
            "Sampling Rate (Hz)",
            "timestep (s)",
            "Units",
        ]
        table.align["Name"] = "l"

        channel_info = []
        for idx, channel in enumerate(self.adicht_data.channels):
            info = {
                "idx": idx,
                "id": channel.id,
                "name": channel.name,
                "records": channel.n_records,
                "samples": channel.n_samples,
                "fs": channel.fs,
                "time_step": channel.dt,
                "units": channel.units,
            }
            channel_info.append(info)
            table.add_row(
                [
                    idx,
                    channel.id,
                    channel.name,
                    channel.n_records,
                    ", ".join(map(str, channel.n_samples)),
                    ", ".join(map(str, channel.fs)),
                    ", ".join(map(str, channel.dt)),
                    channel.units,
                ]
            )
        self.metadata = channel_info
        self.metadata_table = table
        return channel_info

    def print_metadata(self) -> None:
        """Print the metadata table for the channels in the ADInstruments file.

        Extracts and provides a tabular overview of the channels, samples,
        records, sampling rates, units, and time step.

        :return: List of metadata dicts per channel
        :rtype: list[dict]
        """
        _ = self.generate_metadata()
        print("Available channels and metadata:")  # noqa: T201
        print(self.metadata_table)  # noqa: T201

    def get_labels(
        self,
        channel_idxs: list[int] | None = None,
        channel_ids: list[str] | None = None,
    ) -> list[str]:
        """Returns a list of channel names based on a list of channel indices.

        :param channel_idxs: List of channel indices
        :type channel_idxs: list[int]
        :param channel_ids: List of channel IDs to retrieve the labels for
        :type channel_ids: list

        :return: List of channel names.
        :rtype: list[str]
        """
        _channel_idxs = self._get_channel_id(
            channel_ids=cast("list[int] | None", channel_ids), channel_idxs=channel_idxs
        )
        return [self.adicht_data.channels[idx].name for idx in _channel_idxs]

    def _get_channel_id(
        self,
        channel_ids: list[int] | None = None,
        channel_idxs: list[int] | None = None,
    ) -> list[int]:

        if channel_ids is not None and channel_idxs is None:
            return [
                get_dict_key_where_value(self.channel_map, _id) for _id in channel_ids
            ]
        if channel_ids is None and channel_idxs is not None:
            return channel_idxs
        if channel_ids is None and channel_idxs is None:
            msg = "Either channel_id or channel_idx must be set."
            raise ValueError(msg)
        return []

    def _get_record_id(
        self,
        record_id: int | None = None,
        record_idx: int | None = None,
    ) -> int:

        if record_id is not None and record_idx is None:
            return get_dict_key_where_value(self.record_map, record_id)
        if record_id is None and record_idx is not None:
            return record_idx
        if record_id is None and record_idx is None:
            msg = "Either record_id or record_idx must be set."
            raise ValueError(msg)
        return 0

    def get_units(
        self,
        channel_idxs: list[int] | None = None,
        record_idx: int | None = None,
        channel_ids: list[int] | None = None,
        record_id: int | None = None,
    ) -> list[str]:
        """Returns the units for the specified channels and record.

        Returns a list of units based on a list of channel indices and a record
        id. Either channel_idxs or channel_ids must be set, and either record_idx
        or record_id must be set.


        Args:
            channel_idxs (list[int], optional): list of channel indices
            channel_ids (list[int], optional): list of channel IDs to retrieve
                the labels for
            record_idx (int, optional): the record index to retrieve the units for
            record_id (int, optional): the record ID to retrieve the units for

        Returns:
            list[str]: list of units
        """
        _channel_idxs = self._get_channel_id(
            channel_ids=channel_ids,
            channel_idxs=channel_idxs,
        )
        _record_idx = self._get_record_id(record_id=record_id, record_idx=record_idx)
        return [
            self.adicht_data.channels[idx].units[_record_idx] for idx in _channel_idxs
        ]

    def resample_channel(
        self,
        fs_target: int,
        channel_idx: list[int] | int | None = None,
        record_idx: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Resample the specified channel and add a row to the resampled DataFrame.

        Resample the specified channel using a linear interpolation
        method and adds an additional row to the resampled DataFrame.
        Either record_idx or record_id must be set.

        Args:
            channel_idx (list[int], optional): list of channel indices to resample
            record_idx (int, optional): the record index to resample
            fs_target (int): the target sampling rate in Hz
            kwargs: additional arguments to specify the channel ID or record ID
                    instead of indices
                channel_id (int, optional): the channel ID to resample
                record_id (int, optional): the record ID to resample
        Returns:
            pd.DataFrame: Record DataFrame with resampled data for the specified idx
        """
        _channel_idx = self._get_channel_id(
            channel_ids=kwargs.get("channel_id"),
            channel_idxs=cast("list[int] | None", channel_idx),
        )[0]
        _record_idx = self._get_record_id(
            record_id=kwargs.get("record_id"),
            record_idx=record_idx,
        )
        # Extract data for the specific channel
        if _channel_idx not in self.channel_map:
            msg = f"Channel ID '{_channel_idx}' is invalid."
            raise ValueError(msg)

        # Load channel data and Retrieve sampling details: Original time step
        _channel = self.adicht_data.channels[_channel_idx]
        if fs_target == 1 / _channel.dt[_record_idx]:
            msg = "target_rate equals current_rate"
            raise UserWarning(msg)

        # Create DataFrame and set time index
        df = pd.DataFrame(
            {_channel.name: _channel.get_data(self.record_map[_record_idx])}
        )
        df.index = pd.to_timedelta(df.index * _channel.dt[_record_idx], unit="s")

        # New interval based on target rate
        dt_target_timedelta = pd.to_timedelta(1 / fs_target, unit="s")

        fs_original = _channel.fs[_record_idx]
        n_samples_target = int(
            _channel.n_samples[_record_idx] * (fs_target / fs_original)
        )

        # Create an empty DataFrame with target sample rate
        timedelta_index = pd.to_timedelta(
            np.arange(n_samples_target) * dt_target_timedelta.value
        )
        empty_df = pd.DataFrame(
            index=timedelta_index, columns=pd.Index([_channel.name])
        )
        empty_df[_channel.name] = np.nan

        # Merge DataFrames
        df_combined = empty_df.combine_first(df)
        df_combined = df_combined.interpolate(method="linear")
        return df_combined.resample(dt_target_timedelta).interpolate(method="linear")

    def extract_data(
        self,
        channel_idxs: list[int] | None = None,
        record_idx: int | None = None,
        resample_channels: dict[int, int] | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, int]:
        """Extract channel data from specified channels and record.

        Extract channel data from specified channels and record. Optionally,
        resample specified channels to equalize sampling rates across channels.
        Resampling all channels to a different, not yet used, rate is not
        supported. There must be at least one channel that already has the
        target rate and is not specified in the resampling.

        Args:
            channel_idxs (list[int], optional): List of channel indices to
                extract data from.
            record_idx (int, optional): The record index to extract the data from.
            resample_channels (dict[int, int], optional): Dictionary specifying
                channels to resample and their target rates.
                Example: {1: 2000, 3: 2000} - Resample ch 1 and ch 3 to 2000 Hz
            kwargs: Additional arguments to specify the channel IDs or record IDs
                instead of indices.
                channel_ids (list[int], optional): List of channel IDs to extract
                    data from.
                record_id (int, optional): The record ID to extract the data from.

        Returns:
            tuple: A tuple containing:
                - A pandas DataFrame with the extracted and resampled data.
                - The sampling rate (in Hz) of the leading channel.

        """
        _channel_idxs = self._get_channel_id(
            channel_ids=kwargs.get("channel_ids"),
            channel_idxs=channel_idxs,
        )
        _record_idx = self._get_record_id(
            record_id=kwargs.get("record_id"),
            record_idx=record_idx,
        )

        fs_out = []
        data_dict = {}
        non_resampled_channels = []
        for idx in _channel_idxs:
            if idx not in self.channel_map:
                msg = f"Channel idx '{idx}' is invalid."
                raise ValueError(msg)

            if resample_channels and idx in resample_channels:
                resampled_df = self.resample_channel(
                    resample_channels[idx], idx, record_idx
                )

                for column in resampled_df.columns:
                    data_dict[column] = resampled_df[column].values
                fs_out.append(resample_channels[idx])
            else:
                np_data = self.adicht_data.channels[idx].get_data(
                    self.record_map[_record_idx]
                )
                channel_name = self.adicht_data.channels[idx].name
                data_dict[channel_name] = np_data
                non_resampled_channels.append(idx)
                fs_out.append(self.metadata[idx]["fs"][0])

        if len(set(fs_out)) > 1:
            msg = "Output channels have different sampling rates."
            raise ValueError(msg)

        df = pd.DataFrame(data_dict)
        # Select an unsampled channel to read out target sampling
        leader_channel = self.adicht_data.channels[non_resampled_channels[0]]
        df.index = pd.to_timedelta(df.index * leader_channel.dt[_record_idx], unit="s")

        return df, int(leader_channel.fs[_record_idx])
