"""
Copyright 2025 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Type, List
import h5py
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import numpy as np

from compox.algorithm_utils.io_schemas import DataSchema
from compox.internal.hdf5_io import HDF5IO
from compox.server_utils import generate_uuid


class TempStore:
    """
    Temporary store for HDF5 files.

    This utility creates an isolated temporary directory for the lifetime of the
    instance (or context-manager block) and provides convenience methods to
    atomically save/load Pydantic-validated dictionaries to/from HDF5 files.
    Paths are hardened to ensure all I/O stays under the temp root.

    Attributes
    ----------
    root : Path
        Absolute path to the temporary root directory created on init.
    _atomic : bool
        If True, writes are performed to a ``.tmp`` file and atomically
        moved into place with ``os.replace`` upon success.
    """

    def __init__(
        self,
        *,
        base_dir: str | Path | None = None,
        prefix: str = "tmpstore-",
        atomic_writes: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        base_dir : str | Path | None, optional
            Base directory under which the temporary root will be created.
            If ``None``, the system default temp directory is used.
        prefix : str, optional
            Prefix for the created temporary directory name (via ``mkdtemp``).
        atomic_writes : bool, optional
            If True, perform atomic writes (write to ``.tmp`` then rename).
        """
        self._atomic = atomic_writes
        self.root = Path(
            tempfile.mkdtemp(
                prefix=prefix, dir=str(base_dir) if base_dir else None
            )
        )

    def __enter__(self) -> "TempStore":
        """
        Enter the context manager.

        Returns
        -------
        TempStore
            This instance.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Exit the context manager and clean up the temporary directory.

        Parameters
        ----------
        exc_type :
            Exception type if an exception occurred, else ``None``.
        exc :
            Exception instance if an exception occurred, else ``None``.
        tb :
            Traceback if an exception occurred, else ``None``.
        """
        self.cleanup()

    def cleanup(self) -> None:
        """
        Recursively remove the temporary root directory.

        Notes
        -----
        Safe to call multiple times; missing directories are ignored.
        """
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def _final_path(self, base_dir: str | Path, file_id: str) -> Path:
        """
        Compute the absolute on-disk path for a file under the temp root.

        Ensures the returned path is under ``self.root`` and that parent
        directories exist.

        Parameters
        ----------
        base_dir : str | Path
            Subdirectory (relative to ``self.root``) in which the file resides.
        file_id : str
            File identifier or filename.

        Returns
        -------
        Path
            Absolute path to the file inside the temp root.
        """
        p = self.root / base_dir / file_id
        p = self._under_root(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _under_root(self, p: Path) -> Path:
        """
        Validate that ``p`` (resolved) is located under ``self.root``.

        Parameters
        ----------
        p : Path
            Candidate path (can be relative) to validate.

        Returns
        -------
        Path
            The resolved absolute path.

        Raises
        ------
        ValueError
            If the resolved path escapes the temporary root.
        """
        rp = (self.root / p).resolve()
        root = self.root.resolve()
        try:
            rp.relative_to(root)
        except ValueError:
            raise ValueError(f"Path escapes temp root: {p}")
        return rp

    def mkdir(self, path: str | Path) -> Path:
        """
        Create a directory inside the temporary storage.

        Parameters
        ----------
        path : str | Path
            Relative path (under ``self.root``) of the directory to create.

        Returns
        -------
        Path
            Absolute path of the created directory (exists on return).
        """
        dir_path = self.root / path
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def save(
        self,
        path: str | Path,
        files: list[dict],
        pydantic_data_schema: Type[DataSchema],
        parallel: bool = False,
    ) -> list[str]:
        """
        Save dictionaries (validated by a Pydantic schema) as HDF5 files.

        Each input dict is validated using ``pydantic_data_schema`` and written
        to a separate HDF5 file under ``path`` using the corresponding
        ``file_id``. When ``atomic_writes`` is enabled, data is written to a
        temporary ``.tmp`` file and atomically moved into place on success.

        Parameters
        ----------
        path : str | Path
            Subdirectory (under ``self.root``) in which to store the files.
        files : list[dict]
            Payloads to validate and persist (one dict per target file).
        pydantic_data_schema : Type[DataSchema]
            Pydantic model (class) used to validate/normalize each payload.
        parallel : bool, optional
            If True, perform saves using a thread pool.

        Returns
        -------
        list[str]
            Absolute paths to the saved HDF5 files (as strings).

        Raises
        ------
        ValueError
            If ``files`` and ``file_ids`` have different lengths, or if
            validation yields an empty payload for a given ``file_id``.
        Exception
            Propagates any lower-level I/O or validation exceptions.

        """

        def save_one(payload: dict, base_dir: str | Path) -> str:
            validated = pydantic_data_schema.model_validate(
                payload
            ).model_dump()

            fid = generate_uuid()

            if not validated:
                raise ValueError(
                    f"Nothing to save for file_id {fid}, expected keys:{pydantic_data_schema.model_fields.keys()}, got: {payload.keys()}",
                )
            dst = self._final_path(base_dir, fid)
            tmp = dst.with_suffix(dst.suffix + ".tmp") if self._atomic else dst

            # write
            with h5py.File(tmp, "w") as f:
                for key, val in validated.items():
                    if val is None:
                        logger.warning(
                            f"Value for key '{key}' is None, skipping."
                        )
                        continue
                    HDF5IO.write_dataset(f, key, val)
            if self._atomic:
                os.replace(tmp, dst)
            return str(dst)

        try:
            if parallel:
                with ThreadPoolExecutor() as ex:
                    return list(
                        ex.map(
                            lambda args: save_one(*args),
                            zip(files, [path] * len(files)),
                        )
                    )
            else:
                return [save_one(r, path) for r in files]
        except Exception as e:
            logger.exception(f"Failed to save HDF5 data: {e}")
            raise

    def load(
        self,
        paths: List[str] | List[Path],
        parallel: bool = False,
        *keys: str,  # optional: pick specific datasets
    ) -> list[dict]:
        """
        Load HDF5 files into dictionaries validated by a Pydantic schema.

        Each file identified by ``file_ids`` is read from ``path`` (under
        ``self.root``). If ``keys`` are provided, only those datasets are
        loaded (missing keys will be present in the output dict with value
        ``None``); otherwise, all datasets are loaded. The resulting dict
        is validated via ``pydantic_data_schema`` before being returned.

        Parameters
        ----------
        paths : str | Path
            Subdirectory (under ``self.root``) from which to load files.
        pydantic_data_schema : Type[DataSchema]
            Pydantic model (class) to validate/normalize loaded data.
        parallel : bool, optional
            If True, perform loads using a thread pool.
        *keys : str
            Optional dataset names to load selectively.

        Returns
        -------
        list[dict]
            One validated dictionary per input ``file_id``.

        Raises
        ------
        Exception
            Propagates lower-level I/O or validation exceptions.
        """

        def load_one(path: str | Path) -> dict:
            data: dict[str, object] = {}
            with h5py.File(path, "r") as f:
                if keys:
                    for k in keys:
                        data[k] = (
                            HDF5IO.read_dataset(f[k]) if k in f else None
                        )
                else:
                    for k in f.keys():
                        data[k] = HDF5IO.read_dataset(f[k])

            return data

        try:
            if parallel:
                with ThreadPoolExecutor() as ex:
                    return list(
                        ex.map(
                            lambda args: load_one(*args),
                            zip(paths),
                        )
                    )
            else:
                return [load_one(p) for p in paths]
        except Exception as e:
            logger.exception(f"Failed to load HDF5 data: {e}")
            raise


if __name__ == "__main__":
    import numpy as np

    class MyDataSchema(DataSchema):
        a: np.ndarray
        b: int
        c: float

    with TempStore() as store:
        store.mkdir("a")
        store.mkdir("b/c")
        print(store.root)
        paths = store.save(
            "a",
            [
                {"a": np.ones((256, 256)), "b": 2, "c": 3.0},
                {"a": np.zeros((128, 128)), "b": 5, "c": 6.0},
            ],
            MyDataSchema,
            parallel=True,
        )
        print(paths)
        loaded = store.load(
            paths,
            parallel=True,
        )
        assert loaded[0]["a"].shape == (256, 256)
        assert loaded[0]["b"] == 2
        assert loaded[0]["c"] == 3.0
        print(loaded)
