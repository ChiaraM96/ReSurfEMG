"""Sanity tests for the config functions"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from resurfemg.data_connector.config import Config, find_project_root


class TestConfig(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # resolve() expands Windows short names (e.g. MANINE~1), matching
        # the resolved paths Config returns
        self.td = Path(tmp.name).resolve()
        self.root = self.td / "root"

    def write_config(self, raw_config: dict) -> Path:
        config_file = self.td / "config.json"
        config_file.write_text(json.dumps(raw_config), encoding="utf-8")
        return config_file

    # --- loading ---
    def test_roots_only(self):
        self.root.mkdir()
        config_file = self.write_config({"root_data": str(self.root)})
        config = Config(location=config_file)
        self.assertEqual(config.get_directory("root_data"), self.root)

    def test_location_as_str(self):
        self.root.mkdir()
        config_file = self.write_config({"root_data": str(self.root)})
        config = Config(location=str(config_file))
        self.assertEqual(config.get_directory("root_data"), self.root)

    def test_missing_config_path(self):
        with self.assertRaises(ValueError):
            Config(location=self.td / "non_existent" / "config.json")

    def test_raw_config_not_mutated(self):
        self.root.mkdir()
        raw_config = {"root_data": str(self.root), "output_data": "./out"}
        config = Config(location=self.write_config(raw_config))
        self.assertEqual(config._raw, raw_config)

    # --- root_data ---
    def test_missing_root(self):
        config_file = self.write_config({})
        with self.assertRaises(ValueError) as cm:
            Config(location=config_file)
        self.assertIn("root_data", str(cm.exception))

    def test_relative_root(self):
        """A relative root_data resolves against the config file directory."""
        self.root.mkdir()
        config = Config(location=self.write_config({"root_data": "./root"}))
        self.assertEqual(config.get_directory("root_data"), self.root)

    def test_non_existent_root(self):
        config_file = self.write_config({"root_data": str(self.root)})
        with self.assertRaises(FileNotFoundError) as cm:
            Config(location=config_file)
        msg = str(cm.exception).replace("\n", "")
        self.assertIn("root_data", msg)
        self.assertIn("force=True", msg)
        self.assertFalse(self.root.exists())  # nothing created without force

    def test_create_non_existent_root(self):
        config_file = self.write_config({"root_data": str(self.root)})
        config = Config(location=config_file, force=True)
        self.assertTrue(self.root.is_dir())
        self.assertEqual(config.get_directory("root_data"), self.root)

    def test_root_is_a_file(self):
        self.root.touch()
        config_file = self.write_config({"root_data": str(self.root)})
        with self.assertRaises(NotADirectoryError):
            Config(location=config_file)

    # --- other directories ---
    def test_relative_subdirectory(self):
        """Paths starting with "." resolve against root_data."""
        self.root.mkdir()
        config = Config(
            location=self.write_config(
                {"root_data": str(self.root), "output_data": "./out"}
            )
        )
        self.assertEqual(config.get_directory("output_data"), self.root / "out")
        self.assertIn("output_data", config.relative_paths)

    def test_parent_relative_subdirectory(self):
        """'../' must go up from root_data, not be glued onto it."""
        self.root.mkdir()
        config = Config(
            location=self.write_config(
                {"root_data": str(self.root), "output_data": "../sibling"}
            )
        )
        self.assertEqual(config.get_directory("output_data"), self.td / "sibling")

    def test_absolute_override(self):
        self.root.mkdir()
        elsewhere = self.td / "elsewhere"
        config = Config(
            location=self.write_config(
                {"root_data": str(self.root), "output_data": str(elsewhere)}
            )
        )
        self.assertEqual(config.get_directory("output_data"), elsewhere)
        self.assertNotIn("output_data", config.relative_paths)
        self.assertNotIn("output_data", config.missing_paths)

    def test_defaults_backfilled(self):
        self.root.mkdir()
        config = Config(location=self.write_config({"root_data": str(self.root)}))
        # test_data is excluded: it points to the repo's test_data when the
        # tests run inside the ReSurfEMG repository
        expected = {
            "patient_data": self.root / "patient_data",
            "simulated_data": self.root / "simulated",
            "preprocessed_data": self.root / "preprocessed",
            "output_data": self.root / "output",
        }
        for key, path in expected.items():
            self.assertEqual(config.get_directory(key), path, key)
            self.assertIn(key, config.missing_paths)

    def test_all_values_are_paths(self):
        self.root.mkdir()
        config = Config(
            location=self.write_config(
                {"root_data": str(self.root), "output_data": "./out"}
            )
        )
        for key, value in config._loaded.items():
            self.assertIsInstance(value, Path, key)

    def test_invalid_path_value(self):
        self.root.mkdir()
        config_file = self.write_config(
            {"root_data": str(self.root), "output_data": ""}
        )
        with self.assertRaises(ValueError) as cm:
            Config(location=config_file)
        self.assertIn("output_data", str(cm.exception))

    # --- get_directory ---
    def test_get_directory_fallback_when_missing(self):
        self.root.mkdir()
        config = Config(location=self.write_config({"root_data": str(self.root)}))
        fallback = self.td / "fallback"
        self.assertEqual(config.get_directory("not_configured", fallback), fallback)

    def test_get_directory_ignores_fallback_when_configured(self):
        self.root.mkdir()
        config = Config(location=self.write_config({"root_data": str(self.root)}))
        self.assertEqual(
            config.get_directory("root_data", self.td / "fallback"), self.root
        )

    # --- configure ---
    def test_configure_creates_config(self):
        config_file = self.td / "config.json"
        config = Config(location=config_file, configure=True, force=True)
        self.assertTrue(config_file.is_file())
        saved_root = Path(json.loads(config_file.read_text())["root_data"])
        self.assertTrue(saved_root.is_absolute())
        self.assertTrue(config.get_directory("root_data").is_dir())

    def test_configure_refuses_overwrite(self):
        self.root.mkdir()
        config_file = self.write_config({"root_data": str(self.root)})
        with self.assertRaises(FileExistsError):
            Config(location=config_file, configure=True)


class TestFindProjectRoot(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.project_root = Path(tmp.name) / "project"
        self.sub_dir = self.project_root / "subdir"
        self.sub_dir.mkdir(parents=True)

    def test_find_project_root(self):
        (self.project_root / "requirements.txt").touch()
        found_root = find_project_root(current_dir=self.sub_dir)
        self.assertIsNotNone(found_root)
        self.assertTrue(self.project_root.samefile(found_root))

    def test_find_non_existent_project_root(self):
        found_root = find_project_root(current_dir=self.sub_dir)
        self.assertIsNone(found_root)

    def test_find_non_existent_marker_file(self):
        found_root = find_project_root(
            current_dir=self.sub_dir, marker_file="non_existent_file.txt"
        )
        self.assertIsNone(found_root)
