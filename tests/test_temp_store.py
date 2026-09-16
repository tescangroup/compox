"""
Copyright 2025 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import pytest
import numpy as np
import os
import h5py

from compox.training.TempStore import TempStore
from compox.algorithm_utils.io_schemas import DataSchema


class MySchema(DataSchema):
    array: np.ndarray


class MyStringListSchema(DataSchema):
    names: list[str]


def test_temp_store_creation_and_deletion():
    """Test creating and deleting a temporary storage."""
    with TempStore() as temp_store:
        assert os.path.exists(
            temp_store.root
        ), "Temporary storage path does not exist."

    assert not os.path.exists(
        temp_store.root
    ), "Temporary storage path was not deleted."


def test_mkdir():
    """Test creating directories in the temporary storage."""
    with TempStore() as temp_store:
        _ = temp_store.mkdir("subdir1")
        assert os.path.exists(
            os.path.join(temp_store.root, "subdir1")
        ), "Directory was not created."
        _ = temp_store.mkdir("subdir1/subdir2")
        assert os.path.exists(
            os.path.join(temp_store.root, "subdir1", "subdir2")
        ), "Directory was not created."


def test_save_file_and_load_file():
    """Test saving and loading files in the temporary storage."""
    with TempStore() as temp_store:
        my_files = [
            {"array": np.random.rand(10, 10).astype(np.float32)},
            {"array": np.random.rand(20, 20).astype(np.float32)},
        ]

        file_paths = temp_store.save(
            "my_data", my_files, MySchema, parallel=True
        )

        assert len(file_paths) == 2, "Number of saved files does not match."

        assert os.path.exists(
            file_paths[0]
        ), "First file was not saved correctly."
        assert os.path.exists(
            file_paths[1]
        ), "Second file was not saved correctly."
        loaded_files = temp_store.load(file_paths, parallel=True)
        assert len(loaded_files) == 2, "Number of loaded files does not match."
        np.testing.assert_array_equal(
            loaded_files[0]["array"], my_files[0]["array"]
        )
        np.testing.assert_array_equal(
            loaded_files[1]["array"], my_files[1]["array"]
        )

    assert not os.path.exists(
        temp_store.root
    ), "Temporary storage path was not deleted."


def test_save_and_load_string_list():
    with TempStore() as temp_store:
        my_files = [
            {"names": ["mid_intensity", "high_intensity"]},
            {"names": ["foreground"]},
        ]

        file_paths = temp_store.save(
            "my_names", my_files, MyStringListSchema, parallel=False
        )
        loaded_files = temp_store.load(file_paths, parallel=False)

        assert loaded_files == my_files
