"""
 Copyright 2026 TESCAN GROUP, a.s.
All rights reserved
"""

from typing import Any

import h5py
import numpy as np


class HDF5IO:
    """
    Shared helpers for reading and writing simple Python/NumPy values to HDF5.
    """

    @staticmethod
    def read_dataset(dset: h5py.Dataset) -> Any:
        """
        Read an HDF5 dataset and decode UTF-8 string datasets when possible.
        """
        try:
            value = dset.asstr()[()]
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value
        except (TypeError, ValueError, UnicodeDecodeError):
            value = dset[()]
            return (
                value.decode("utf-8")
                if isinstance(value, (bytes, bytearray, np.bytes_))
                else value
            )

    @staticmethod
    def write_dataset(handle: h5py.File, key: str, value: Any) -> None:
        """
        Write a Python value into an HDF5 dataset, including string lists.
        """
        if isinstance(value, str):
            dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset(
                key,
                data=np.array(value, dtype=object),
                dtype=dtype,
            )
            return

        if isinstance(value, (list, tuple)) and all(
            isinstance(item, str) for item in value
        ):
            dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset(
                key,
                data=np.asarray(value, dtype=object),
                dtype=dtype,
            )
            return

        if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S"}:
            dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset(
                key,
                data=value.astype(object),
                dtype=dtype,
            )
            return

        handle.create_dataset(key, data=value)
