"""Copyright 2022 Netherlands eScience Center and Twente University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods to let the user configure all paths for data
instead of hard-coding them, as well as methods to check data integrity.
The data integrity can be checked because this file contains hash functions
to track data. Synthetic data can be made with several methods.
"""  # noqa: D205

import hashlib
import json
import logging
import os
import textwrap
from importlib.resources import files
from pathlib import Path
from typing import ClassVar

import pandas as pd

logger = logging.getLogger(__name__)
CURRENT_DIR = Path.cwd()
BUF_SIZE = 65536


def convert_to_os_path(
    path: str,
) -> str:
    """This function converts a path to a os readable path.

    Args:
        path(str): The path to convert.

    Returns:
        str: The converted path.
    """
    return path.replace(os.sep if os.altsep is None else os.altsep, os.sep)


def find_project_root(
    current_dir: Path = CURRENT_DIR,
    marker_file: str | None = None,
    prefer_build_markers: bool = True,
) -> Path | None:
    """Walks upward from `current_dir` to find a plausible Python project root.

    Returns the path if found, else None. If `marker_file` is provided,
    it is uses the `marker_file` in favor of looking for standard build/
    VCS markers. If `prefer_build_markers` is True, prioritize pyproject/setup
    files over .git. Markers checked are:
    Build-markers: pyproject.toml, setup.cfg, setup.py, poetry.lock,
    Pipfile, requirements.txt, tox.ini
    VCS-markers: .git

    Args:
        current_dir (Path): The directory to start searching from.
        marker_file (Optional[str]): Look for a specific marker file.
        prefer_build_markers (bool): Whether to prefer build markers over VCS
            markers.

    Returns:
        Optional[Path]: The absolute path to the root directory of the repository.
    """
    if isinstance(current_dir, str):
        current_dir = Path(current_dir)
    start = current_dir.resolve()
    if marker_file is not None:
        build_markers = {marker_file}
        vcs_markers = set()
        prefer_build_markers = True
    else:
        build_markers = {
            "pyproject.toml",
            "setup.cfg",
            "setup.py",
            "poetry.lock",
            "Pipfile",
            "requirements.txt",
            "tox.ini",
        }
        vcs_markers = {".git"}

    candidate_by_priority = None

    for current in [start, *start.parents]:
        items = set()
        if current.exists():
            items = {p.name for p in current.iterdir()}

        # Build-markers = Check intersection between items and build markers
        if prefer_build_markers and items & build_markers:
            return current

        # Record .git as a candidate (don't return immediately if preferring
        # build markers)
        if marker_file is None and items & vcs_markers:
            candidate_by_priority = candidate_by_priority or current

    # If we didn't find a build marker, use .git candidate
    if candidate_by_priority:
        return candidate_by_priority
    if marker_file is not None:
        print(f"Marker file '{marker_file}' not found in any parent directory")
    return None


class Config:
    """Configuration class for ResurfEMG data paths.

    A configuration class for ResurfEMG data paths, allowing to specify
    paths in a JSON file instead of hardcoding them within the code.
    A typical workflow would be:
    1. Create a Config object, which loads the configuration file (config.json)
    from the provided location or from default locations:
    config = Config(verbose=True)
    2. Extract a configured directory using the get_directory method:
    base_path = config.get_directory('root_data').

    If no location is specified it will try to load the configuration file from
    the default locations in the following order:
    - ./config.json                 (current working directory)
    - PROJECT_ROOT/config.json      (project root directory, if located)
    - ~/.resurfemg/config.json      (user config in Windows)
    - /etc/resurfemg/config.json    (system-wide config in Linux or OSX)

    If 'configure' is set to True, a new configuration file will be created
    at the first available location in the above list.
    If 'force' is set to True, any existing configuration file will be
    overwritten.

    For details on how to set up the configuration file, see:
    - print(config.usage()): instructions on where to create a config file.
    - print(config.config_file_help()): details on the config file structure.
    -----------------------------------------------------------------------
    """

    required_directories: ClassVar[list[str]] = ["root_data"]

    def __init__(
        self,
        location: Path | None = None,
        configure: bool = False,
        verbose: bool = False,
        force: bool = False,
    ):
        """Initialize the Config object.

        Args:
            location (Path | None): Path to the configuration file. If None, default
                locations will be searched.
            configure (bool): If True, a new configuration file will be created at the
                first available location. Default is False.
            verbose (bool): If True, prints the loaded configuration. Default is False.
            force (bool): If True, any existing configuration file will be overwritten.
                Default is False.

        Raises:
            ValueError: If the configuration file is not found.
        """
        self._raw: dict[str, str]
        self._loaded: dict[str, Path]
        self.repo_root = find_project_root()
        self.relative_paths = []
        self.missing_paths = []
        self.default_layout = self.get_default_layout()
        if isinstance(location, str):
            location = Path(location)
        if configure:
            location = self.setup_config(location=location, force=force)
        _path = self.load(location, verbose=verbose)
        self.parse_paths(_path)
        if configure:
            print(f"Created config. See and edit it at:\n {_path}\n")
        if verbose or configure:
            print("The following name-path combinations are configured:")
            self.print_config()
        self.validate(_path, force=force)

    def usage(self) -> str:
        """Provide usage instructions for configuring paths.

        Provide feedback if the paths are not configured or not configured
        correctly. It contains instructions on how to configure the paths.
        -----------------------------------------------------------------------
        """
        return textwrap.dedent(
            """
            At initialization, the Config class looks for a configuration
            file named "config.json" at 'location' (if specified), or
            in one of the following locations (in search order):

            {}

            A "config.json" can be created automatically by running:

                ```
                config = Config(configure=True)
                ```

            This will create a "config.json" file in the first available of:
            - 'location', if provided
            - the project root, if found
            - the current working directory

            Alternatively, you can create a "config.json" file manually at any
            of above listed locations.

            For more information on how to set up the "config.json" file,
            please refer to the following help section:
                ```
                Config.config_file_help()
                ```
            """
        ).format("\n".join(str(path) for path in self.default_locations))

    def config_file_help(self) -> str:
        """This function provides help on how to set up the configuration file."""
        return textwrap.dedent(
            """
            The contents of "config.json" should specify at least the
            "root_data" directory:

            {{
                "root_data": "{}"
            }}

            The directory layout is expected to be based on the "root_data"
            directory with subdirectories for different types of data.
            "root_data" is required to exist. The default directories
            "patient_data", "simulated_data", "preprocessed_data",
            "output_data", and "test_data" are optional to be included in the
            config.json file. If missing, they will be added to the returned
            config relative to "root_data".

            You can override any individual directory (or subdirectory) by
            specifying it in the config.json file.
            """
        ).format("/path/to/storage")

    def get_default_layout(self) -> dict:
        """This function returns the default layout of the configuration file.

        Returns:
            dict: The default layout of the configuration file.
        """
        if self.repo_root is not None:
            # In the ResurfEMG project, the test data is stored in ./test_data
            test_path = self.repo_root / "test_data"
            test_data_path = str(test_path) if test_path.is_dir() else "{}/test_data"
            self.default_locations = (
                Path.cwd() / "config.json",
                self.repo_root / "config.json",
                Path.home() / ".resurfemg" / "config.json",
                Path("/etc/resurfemg/config.json"),
            )
        else:
            test_data_path = "{}/test_data"
            self.default_locations = (
                Path.cwd() / "config.json",
                Path.home() / ".resurfemg" / "config.json",
                Path("/etc/resurfemg/config.json"),
            )
        return {
            "root_data": "{}/not_pushed",
            "test_data": test_data_path,
            "patient_data": "{}/patient_data",
            "simulated_data": "{}/simulated",
            "preprocessed_data": "{}/preprocessed",
            "output_data": "{}/output",
        }

    def load(self, location: Path | None = None, verbose: bool = False) -> Path:
        """Load the configuration file.

        This function loads the configuration file. If no location is specified
        it will try to load the configuration file from the default locations:
        - ./config.json
        - ~/.resurfemg/config.json
        - /etc/resurfemg/config.json
        - PROJECT_ROOT/config.json.

        Args:
            location (Path | None): The location of the configuration file. If None,
                default locations will be searched.
            verbose (bool): If True, prints the loaded configuration. Default is False.

        Returns:
            Path: The path of the loaded configuration file.

        Raises:
            ValueError: If the configuration file is not found.
        """
        locations = [Path(location)] if location is not None else self.default_locations
        _failed_paths = []
        for _path in locations:
            try:
                with _path.open() as f:
                    self._raw = json.load(f)
            except FileNotFoundError:
                _failed_paths.append(_path)
                continue
            if verbose:
                logger.info("Loaded config file from:\n%s", _path)
            break
        else:
            msg = (
                "Config file not found in any of the following locations:\n"
                + "\n".join(str(p) for p in _failed_paths)
                + "\n\n"
                + self.usage()
            )
            raise ValueError(msg)
        return _path

    def parse_paths(self, _path: Path) -> None:
        """Parse and resolve the paths in the configuration file.

        A relative "root_data" is resolved against the directory of the config
        file. Other paths starting with "." are resolved against "root_data".
        Optional directories missing from the config file are back-filled from
        the default layout relative to "root_data".

        Args:
            _path (Path): Path of the loaded configuration file.

        Raises:
            ValueError: If a required directory is missing from the config file.
        """
        # Check if user specified all required directories.
        for directory in self.required_directories:
            if directory not in self._raw:
                msg = (
                    f'Missing required directory "{directory}" in config file.'
                    f"\nThe config file was loaded from:\n {_path}\n"
                    + self.config_file_help()
                )
                raise ValueError(msg)

        # Convert all paths to OS readable paths.
        root = Path(self._raw["root_data"])
        if not root.is_absolute():
            root = (Path(_path).resolve().parent / root).resolve()
        self.missing_paths = sorted(set(self.default_layout) - set(self._raw))
        self.relative_paths = []
        self._loaded = {"root_data": root}
        # User values override the defaults; defaults back-fill the rest.
        # Default values are "{}/..." templates or an absolute test_data path.
        for key, value in {**self.default_layout, **self._raw}.items():
            if key == "root_data":
                continue
            if not isinstance(value, str) or not value:
                msg = f'Invalid path for "{key}" in config file: {value!r}'
                raise ValueError(msg)
            if "{}" in value:
                value = value.format(root)  # noqa: PLW2901
            if value.startswith("."):
                self._loaded[key] = (root / value).resolve()
                self.relative_paths.append(key)
            else:
                self._loaded[key] = Path(value)

    def validate(self, _path: Path, force: bool = False) -> None:
        """Validate the configuration file.

        This function validates the configuration file. It checks if the
        required directories exist.

        Args:
            _path (Path): Path of the loaded configuration file.
            force (bool): If True, create the required directories if they do not exist.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            NotADirectoryError: If a required directory points to a file,
                not a directory.
        """
        for req_dir in self.required_directories:
            dir_path = self._loaded[req_dir]
            if dir_path.is_dir():
                continue
            if dir_path.exists():
                msg = (
                    f'Required directory "{req_dir}" points to a file, not a '
                    f"directory:\n {dir_path}\nEdit the config file at:\n {_path}\n"
                )
                raise NotADirectoryError(msg)
            if force:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"Created required directory at:\n {req_dir}: {dir_path}\n")
                continue
            msg = " ".join(
                [
                    f"Required directory '{req_dir}' specified in the config file",
                    "does not exist. ",
                    "Create it yourself or re-run with `force=True` to create",
                    f"it at:\n {dir_path}\nAlternatively,",
                    f"edit the config file at:\n {_path}\n",
                ]
            )
            raise FileNotFoundError(
                "\n".join(textwrap.fill(line, width=79) for line in msg.splitlines())
            )

    def print_config(self) -> None:
        """This function prints the current configuration."""
        print(79 * "-")
        print(f"  {'Name': <15}\t{'Path': <50}")
        print(79 * "-")
        print(f"  {'root_data': <15}\t{self._loaded['root_data']!s: <50}")
        print(79 * "-")
        for key, path_value in self._loaded.items():
            value = str(path_value)
            if key != "root_data":
                if key in self.missing_paths:
                    _rel_flag = "+ "
                else:
                    _rel_flag = "* " if key in self.relative_paths else "  "
                print(_rel_flag + f"{key: <15}" + f"\t{value: <50}")
        print(79 * "-")
        if len(self.relative_paths) > 0:
            print("* Path is defined relative to root_data.")
        if self.missing_paths is not None and len(self.missing_paths) > 0:
            print(
                "+ Default entry is not defined and has been set"
                " according to default path."
            )

    def get_directory(self, directory: str, value: Path | None = None) -> Path | None:
        """Return a directory from the configuration.

        Args:
            directory (str): Name of the directory, e.g. "output_data".
            value (Path, optional): Fallback returned if the directory is not
                configured. Default is None.

        Returns:
            Path | None: The configured directory, or ``value`` if it is not
                configured.
        """
        if directory in self._loaded:
            return self._loaded[directory]
        print(
            f'Directory "{directory}" not found in config.\n'
            "The following directories are configured:"
        )
        self.print_config()
        return value

    def get_config(self) -> dict[str, Path]:
        """This function returns the configuration file.

        Returns:
            dict: The configuration file.
        """
        return self._loaded

    def setup_config(self, location: Path | None = None, force: bool = False) -> Path:
        """This function sets up the configuration file.

        It creates a config file from an example file if it does not exist.

        Args:
            location (Path | None): The location of the configuration file.
            force (bool): Whether to overwrite the existing configuration file.

        Returns:
            Path: The path to the configuration file.
        """
        if location is not None:
            location = location.resolve()
            base_path = location.parent if location.name == "config.json" else location
        elif self.repo_root is not None:
            base_path = self.repo_root
        else:
            base_path = Path.cwd()

        self.create_config_from_example(base_path, force=force)
        return base_path / "config.json"

    def create_config_from_example(self, location: Path, force: bool = False) -> None:
        """This function creates a config file from an example file.

        Args:
            location (Path): The location where the config file will be created.
            force (bool): Whether to overwrite the existing configuration file.
        """
        example_path = files("resurfemg").joinpath("data_connector/config_example.json")
        config_path = location / "config.json"
        if config_path.is_file() and not force:
            msg = (
                f"Config file already exists at {config_path}."
                " Use `force=True` to overwrite."
            )
            raise FileExistsError(msg)
        example = json.loads(example_path.read_text(encoding="utf-8"))
        # Adjust root_path to an absolute path
        root = Path(example["root_data"])
        if not root.is_absolute():
            example["root_data"] = str((config_path.parent / root).resolve())
        config_path.write_text(json.dumps(example, indent=4), encoding="utf-8")


def hash_it_up_right_all(
    origin_directory: Path | str, file_extension: str
) -> pd.DataFrame:
    """Check the hash of all files in a directory.

    Hashing function to check files are not corrupted or to assure
    files are changed. This function hashes all files in a directory.
    -----------------------------------------------------------------------
    :param origin_directory: The string of the directory with files to hash
    :type origin_directory: str
    :param file_extension: File extension
    :type file_extension: str

    :returns df: The hash values of the files
    :rtype df: pandas.DataFrame
    """
    hash_list = []
    file_names = []
    non_suspects1 = Path(origin_directory).glob("*" + file_extension)

    for file in non_suspects1:
        sha256 = hashlib.sha256()
        with file.open("rb") as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        result = sha256.hexdigest()
        hash_list.append(result)
        file_names.append(str(file))

    df = pd.DataFrame(hash_list, pd.Index(file_names))
    df.columns = ["hash"]
    df = df.reset_index()
    return df.rename(columns={"index": "file_name"})
