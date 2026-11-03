"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to automatically find specified files and folders.
"""  # noqa: D205

from pathlib import Path, PurePath

import pandas as pd

DEFAULT_REGEX = "*"


def _check_valid_path(path: str | Path) -> bool:
    """Check if the provided path is valid."""
    if not isinstance(path, (str, Path)):
        msg = "base_path should be a str or Path."
        raise TypeError(msg)
    if not Path(path).is_dir():
        msg = "Specified path cannot be found."
        raise ValueError(msg)
    return True


def _check_folder_levels(
    folder_levels: list[str] | str | None,
) -> tuple[list[str], int]:
    if folder_levels is None:
        folder_levels = ""
    if not isinstance(folder_levels, (list, str)):
        msg = "folder_levels should be a list or str."
        raise TypeError(msg)
    if isinstance(folder_levels, str):
        folder_levels = [folder_levels] if folder_levels else []
    return list(folder_levels), len(folder_levels)


def find_files(
    base_path: str | Path,
    file_name_regex: str = DEFAULT_REGEX,
    extension_regex: str = DEFAULT_REGEX,
    folder_levels: list[str] | str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """Regex-based file discovery function.

    Find files with the file name and extension according to filename pattern
    `file_name_regex`.`extension_regex` starting from the provided base_path
    according to the provided folder_leves. If `folder_levels` is None, all
    files matching the name pattern included, no matter the data organisation.
    ---------------------------------------------------------------------------

    Args:
        base_path(str): Path to starting directory
        file_name_regex(str, optional): file name pattern, see Python Regex docs
            (Default value = None)
        extension_regex(str, optional): file extension pattern, see Python Regex docs
            (Default value = None)
        folder_levels(list(str) or str, optional): data directory organisation,
            e.g. ['patient', 'date'] (Default value = None)
        verbose(bool, optional): Provide feedback about non-included files
            (Default value = True)

    Returns:
      pd.DataFrame: Matching file paths tabled by the folder_levels

    """
    _check_valid_path(base_path)

    if not isinstance(file_name_regex, str):
        msg = "file_name_regex should be a str."
        raise TypeError(msg)

    if not isinstance(extension_regex, str):
        msg = "extension_regex should be a str."
        raise TypeError(msg)

    folder_levels, depth = _check_folder_levels(folder_levels)
    folder_levels = [*folder_levels, "files"]

    matching_files = Path(base_path).rglob(file_name_regex + "." + extension_regex)

    files_structure = [
        file_name.relative_to(base_path).parts for file_name in matching_files
    ]

    matching_files_structure = []
    non_matching_files_structure = []
    for file in files_structure:
        file_depth = len(file)
        if depth == 0:
            matching_files_structure.append((PurePath(*file).as_posix(),))
        elif file_depth - 1 == depth:
            matching_files_structure.append(file)
        else:
            non_matching_files_structure.append(file)

    files = pd.DataFrame(matching_files_structure, columns=pd.Index(folder_levels))
    if verbose is True and len(non_matching_files_structure) > 0:
        print(
            "These files did not match the provided depth:\n",
            (non_matching_files_structure),
        )
    return files


def find_folders(
    base_path: str | Path,
    folder_levels: list[str] | str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """Find subfolders in a provided path.

    Find folders up to the depth of the provided folder_levels starting from
    the provided base_path. If `folder_levels` is None, all folders in the
    provided are included, no matter the data organisation.
    ---------------------------------------------------------------------------

    Args:
        base_path(Path): Path to starting directory
        folder_levels(list(str) or str, optional): data directory organisation,
            e.g. ['patient', 'date'] (Default value = None)
        verbose(bool, optional): Provide feedback about non-included files
            (Default value = True)

    Returns:
        folders: pd.DataFrame: folders: Folder paths tabled by the folder_levels

    """
    _check_valid_path(base_path)

    folder_levels, depth = _check_folder_levels(folder_levels)
    if depth == 0:
        folder_levels = ["destination"]

    matching_paths = Path(base_path).glob("*/" * max(depth, 1))
    path_structure = []
    path_structure = [
        path_name.relative_to(base_path).parts for path_name in matching_paths
    ]

    matching_path_structure = []
    non_matching_path_structure = []
    for sub_path in path_structure:
        subpath_depth = len(sub_path)
        if subpath_depth == max(depth, 1):
            matching_path_structure.append(sub_path)
        else:
            non_matching_path_structure.append(sub_path)

    folders = pd.DataFrame(matching_path_structure, columns=pd.Index(folder_levels))
    if verbose is True and len(non_matching_path_structure) > 0:
        print(
            "These paths did not match the provided depth:\n",
            (non_matching_path_structure),
        )
    return folders
