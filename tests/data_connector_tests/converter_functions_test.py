"""sanity tests for the converter_functions submodule of the resurfemg library"""

import os
import platform
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

# converter_functions
from resurfemg.data_connector import converter_functions
from resurfemg.data_connector.config import hash_it_up_right_all
from resurfemg.data_connector.converter_functions import load_file, poly5unpad

# tmsisdk_lite
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader

base_path = os.path.join(
    os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
    "test_data",
)


sample_emg_poly5 = os.path.join(
    base_path,
    "emg_data_synth_quiet_breathing.Poly5",
)
sample_emg_mat = os.path.join(
    base_path,
    "emg_data_synth_quiet_breathing.mat",
)
sample_emg_csv = os.path.join(
    base_path,
    "emg_data_synth_quiet_breathing.csv",
)


class TestLoadData(unittest.TestCase):
    def test_load_poly5(self):
        np_array, *_ = converter_functions.load_file(
            file_path=sample_emg_poly5
        )
        self.assertEqual(np_array.shape[0], 2)

    def test_load_mat(self):
        np_array, *_ = converter_functions.load_file(
            file_path=sample_emg_mat,
            key_name="mat5_data",
        )
        self.assertEqual(np_array.shape[0], 2)

    def test_load_csv(self):
        np_array, *_ = converter_functions.load_file(file_path=sample_emg_csv)
        self.assertEqual(np_array.shape[0], 2)


class TestConverterMethods(unittest.TestCase):
    def test_poly5unpad(self):
        reading = Poly5Reader(sample_emg_poly5)
        unpadded = poly5unpad(sample_emg_poly5)
        unpadded_line = unpadded[0]
        self.assertEqual(len(unpadded_line), reading.num_samples)

    def test_Poly5Reader(self):
        reading = Poly5Reader(sample_emg_poly5)
        self.assertEqual(reading.num_channels, 2)


class TestHashMethods(unittest.TestCase):
    def test_hash_it_up_right_all(self):
        tempfile1 = "sample_emg_poly5_t.Poly5"
        tempfile2 = "sample_emg_poly5_t.Poly5"
        with TemporaryDirectory() as td:
            with open(os.path.join(td, tempfile1), "w") as tf:
                tf.write("string")
            with open(os.path.join(td, tempfile2), "w") as tf:
                tf.write("string")
            self.assertTrue(
                hash_it_up_right_all(td, ".Poly5").equals(
                    hash_it_up_right_all(td, ".Poly5")
                )
            )


class TestLoadFile(unittest.TestCase):
    file_path = os.path.join(base_path, "emg_data_synth_quiet_breathing")

    def test_load_file_poly5(self):
        file_name = self.file_path + ".Poly5"
        np_data, df_data, metadata = load_file(file_name)
        assert isinstance(np_data, np.ndarray)
        assert isinstance(df_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert metadata["file_extension"] == "Poly5"

    def test_load_file_adidat(self):
        file_name = self.file_path + ".adidat"
        if platform.system() == "Windows":
            np_data, df_data, metadata = load_file(file_name)
            assert isinstance(np_data, np.ndarray)
            assert isinstance(df_data, pd.DataFrame)
            assert isinstance(metadata, dict)
            assert metadata["file_extension"] == "adidat"
        else:
            with self.assertRaises(UserWarning):
                load_file(file_name)

    def test_load_file_csv(self):
        file_name = self.file_path + ".csv"
        np_data, df_data, metadata = load_file(file_name)
        assert isinstance(np_data, np.ndarray)
        assert isinstance(df_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert metadata["file_extension"] == "csv"

    def test_load_file_mat(self):
        file_name = self.file_path + ".mat"
        np_data, df_data, metadata = load_file(file_name, key_name="mat5_data")
        assert isinstance(np_data, np.ndarray)
        assert isinstance(df_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert metadata["file_extension"] == "mat"

    # def test_load_file_npy(self):
    #     file_name = self.file_path + '.npy'
    #     np_data, df_data, metadata = load_file(file_name)
    #     assert isinstance(np_data, np.ndarray)
    #     assert isinstance(df_data, pd.DataFrame)
    #     assert isinstance(metadata, dict)
    #     assert metadata['file_extension'] == 'npy'
