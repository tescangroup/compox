"""
Copyright 2026 Tescan GROUP, a.s.
All rights reserved
"""

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

from loguru import logger

from compox.database_connection import BaseConnection
from compox.exceptions import (
    CompoxAlgorithmError,
    CompoxNotFoundError,
    CompoxValidationError,
)
from compox.training.AlgorithmCheckpoint import AlgorithmCheckpoint

CHUNK_SIZE = 1024 * 1024


class AlgorithmExporter:
    """
    Export an already deployed algorithm (stored in DB) back into a zip file
    that reconstructs the algorithm directory (module .py files + assets).

    This is the inverse of `AlgorithmDeployer.store_algorithm` on the *stored*
    representation (obfuscated & renamed sources, hashed assets).
    """

    def __init__(
        self,
        database_connection: BaseConnection.BaseConnection,
        algorithm_collection_name: str = "algorithm-store",
        module_collection_name: str = "module-store",
        asset_collection_name: str = "asset-store",
        checkpoint_collection_name: str = "algorithm-checkpoint-store",
    ) -> None:
        self._db = database_connection
        self._alg_coll = algorithm_collection_name
        self._mod_coll = module_collection_name
        self._asset_coll = asset_collection_name
        self._checkpoint_coll = checkpoint_collection_name
        self.logger = logger.bind(log_type="EXPORTER")

        self.logger.info("Algorithm export service initialized")

    def export_algorithm_to_zip(
        self,
        algorithm_name: str,
        algorithm_major_version: str,
        target_zip_path: str,
        algorithm_minor_version: str | None = None,
        algorithm_checkpoint_id: str | None = None,
    ) -> str:
        """
        Export an algorithm with given algorithm_name and version to a zip file.

        Parameters
        ----------
        algorithm_name : str
            The algorithm_name as stored in algorithm-store (the same value that
            `AlgorithmDeployer.store_algorithm` returns).
        algorithm_major_version : str
            The major version of the algorithm.
        target_zip_path : str
            The path to the target zip file to create.
        algorithm_minor_version : str | None, optional
            The minor version of the algorithm. If None, the latest minor
            version is exported, by default None.
        algorithm_checkpoint_id : str | None, optional
            If specified, the assets from the given checkpoint are used to
            override the default assets of the algorithm, by default None.
        Returns
        -------
        str
            The path to the created zip file.
        """
        self.logger.info(
            f"Exporting algorithm '{algorithm_name}' v{algorithm_major_version}.{algorithm_minor_version} to zip..."
        )

        try:
            algorithm_record = self._get_algorithm_record(
                algorithm_name,
                algorithm_major_version,
            )
        except KeyError as ke:
            self.logger.error(
                f"Algorithm '{algorithm_name}' v{algorithm_major_version} not found."
            )
            raise CompoxNotFoundError(
                f"Algorithm '{algorithm_name}' v{algorithm_major_version} not found.",
                code="algorithm_not_found",
                details={
                    "algorithm_name": algorithm_name,
                    "algorithm_major_version": algorithm_major_version,
                },
                cause=ke,
            ) from ke

        if not algorithm_minor_version:
            # get the latest minor version
            exported_minor_version: str = algorithm_record[
                "latest_algorithm_minor_version"
            ]
        else:
            exported_minor_version = algorithm_minor_version

        try:
            minor_version_record = algorithm_record["algorithm_minor_version"][
                str(exported_minor_version)
            ]
        except KeyError as ke:
            self.logger.error(
                f"Algorithm '{algorithm_name}' v{algorithm_major_version}.{exported_minor_version} not found."
            )
            raise CompoxNotFoundError(
                f"Algorithm '{algorithm_name}' v{algorithm_major_version}.{exported_minor_version} not found.",
                code="minor_version_not_found",
                details={
                    "algorithm_name": algorithm_name,
                    "algorithm_major_version": algorithm_major_version,
                    "algorithm_minor_version": exported_minor_version,
                },
                cause=ke,
            ) from ke

        assets_dict: dict[str, str] = minor_version_record.get("assets", {})
        module_id: str = minor_version_record["module_id"]

        if algorithm_checkpoint_id:
            try:
                algorithm_checkpoint = AlgorithmCheckpoint(
                    self._db, checkpoint_id=algorithm_checkpoint_id
                )
            except Exception as e:
                self.logger.error(f"Failed to load algorithm checkpoint: {e}")
                raise CompoxNotFoundError(
                    f"Checkpoint '{algorithm_checkpoint_id}' not found.",
                    code="checkpoint_not_found",
                    details={"checkpoint_id": algorithm_checkpoint_id},
                    cause=e,
                ) from e

            # override the algorithm assets with the ones from the checkpoint
            for (
                key,
                value,
            ) in algorithm_checkpoint.checkpoint_manifest.assets.items():
                try:
                    assets_dict[key] = value
                except Exception as e:
                    self.logger.error(
                        f"Failed to override asset '{key}' from checkpoint: {e}. "
                        "Make sure the checkpoint is compatible with the algorithm."
                    )
                    raise CompoxValidationError(
                        f"Checkpoint '{algorithm_checkpoint_id}' is invalid: failed to override asset '{key}': {e}.",
                        code="invalid_checkpoint",
                        details={
                            "checkpoint_id": algorithm_checkpoint_id,
                            "asset_key": key,
                        },
                        cause=e,
                    ) from e
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            self._extract_module_zip(module_id, tmp_root)
            self._restore_assets(assets_dict, tmp_root)
            target_zip_path = str(Path(target_zip_path).absolute())
            self._make_zip_from_directory(tmp_root, target_zip_path)

        return target_zip_path

    def export_algorithm_zip_stream(
        self,
        algorithm_name: str,
        algorithm_major_version: str,
        algorithm_minor_version: str | None = None,
        algorithm_checkpoint_id: str | None = None,
    ) -> tuple[str, str, Iterator[bytes]]:
        """
        A wrapper around `export_algorithm_to_zip` that returns a streaming
        iterator over the created zip file.
        Parameters
        ----------
        algorithm_name : str
            The algorithm_name as stored in algorithm-store (the same value that
            `AlgorithmDeployer.store_algorithm` returns).
        algorithm_major_version : str
            The major version of the algorithm.
        algorithm_minor_version : str | None, optional
            The minor version of the algorithm. If None, the latest minor
            version is exported, by default None.
        algorithm_checkpoint_id : str | None, optional
            If specified, the assets from the given checkpoint are used to
            override the default assets of the algorithm, by default None.
        Returns
        -------
        tuple[str, str, Iterator[bytes]]
            The filename, path to the created zip file, and an iterator over
            the zip file contents in chunks.
        """
        fd, tmp_zip_path = tempfile.mkstemp(
            suffix=".zip", prefix="algo_export_"
        )
        os.close(fd)
        tmp_zip_path = str(Path(tmp_zip_path).absolute())

        self.export_algorithm_to_zip(
            algorithm_name=algorithm_name,
            algorithm_major_version=algorithm_major_version,
            target_zip_path=tmp_zip_path,
            algorithm_minor_version=algorithm_minor_version,
            algorithm_checkpoint_id=algorithm_checkpoint_id,
        )

        filename = f"{algorithm_name}_v{algorithm_major_version}"

        def iter_file() -> Iterator[bytes]:
            with open(tmp_zip_path, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk

        return filename, tmp_zip_path, iter_file()

    def _get_algorithm_record(
        self,
        algorithm_name: str,
        algorithm_major_version: str,
    ) -> dict:
        """
        Load and parse the algorithm JSON from algorithm-store.

        Parameters
        ----------
        algorithm_name : str
            The name of the algorithm.
        algorithm_major_version : str
            The major version of the algorithm.
        Returns
        -------
        dict
            The parsed algorithm record.
        """
        # simplest approach: list all keys and pick the one that matches prefix
        all_keys = self._db.list_objects(self._alg_coll)

        self.logger.info(
            f"Looking for algorithm record with name '{algorithm_name}' and version {algorithm_major_version} "
            f"in collection '{self._alg_coll}' among {len(all_keys)} objects."
        )
        suffix = f"~{algorithm_name}~{algorithm_major_version}"
        matches = [k["Key"] for k in all_keys if k["Key"].endswith(suffix)]
        if not matches:
            raise KeyError(
                f"No algorithm record found for name '{algorithm_name}' of version "
                f"{algorithm_major_version}"
                f"in collection '{self._alg_coll}'."
            )
        if len(matches) > 1:
            self.logger.warning(
                f"Multiple algorithm records match id '{algorithm_name}' "
                f"and version {algorithm_major_version} "
                f"using the first one: {matches[0]}"
            )

        key = matches[0]
        raw = self._db.get_objects(self._alg_coll, [key])[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise CompoxAlgorithmError(
                f"Algorithm record for key '{key}' is not valid JSON.",
                code="invalid_algorithm_record",
                cause=e,
            ) from e

        return data

    def _extract_module_zip(self, module_id: str, root_dir: Path) -> None:
        """
        Fetch the module zip from module-store and extract it under root_dir.

        Returns the path to the module root directory.
        """
        raw = self._db.get_objects(self._mod_coll, [module_id])[0]
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError(
                f"Module object for id '{module_id}' is not bytes; "
                f"got {type(raw)!r}"
            )

        module_zip_path = root_dir / "module.zip"
        with open(module_zip_path, "wb") as f:
            f.write(raw)

        with zipfile.ZipFile(module_zip_path, "r") as zf:
            for member in zf.infolist():
                parts = Path(member.filename).parts

                # skip top-level directory
                if len(parts) <= 1:
                    continue

                target_path = root_dir.joinpath(*parts[1:])
                self._assert_path_within_root(root_dir, target_path)

                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target_path, "wb") as dst:
                        dst.write(src.read())

        # remove the temporary zip file
        os.remove(module_zip_path)

        return

    def _restore_assets(self, assets: dict[str, str], root_dir: Path) -> None:
        """
        Restore all assets into root_dir using their stored relative paths.

        The keys in `assets` are already normalized (forward slashes etc.)
        via AlgorithmDeployer.process_path_to_dict_key.
        """
        for rel_path, asset_id in assets.items():
            # convert forward-slash path to OS-specific path
            rel_path_os = Path(*rel_path.split("/"))
            target_path = root_dir / rel_path_os
            self._assert_path_within_root(root_dir, target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            raw = self._db.get_objects(self._asset_coll, [asset_id])[0]
            if not isinstance(raw, (bytes, bytearray)):
                raise TypeError(
                    f"Asset object for id '{asset_id}' is not bytes; "
                    f"got {type(raw)!r}"
                )

            with open(target_path, "wb") as f:
                f.write(raw)

    @staticmethod
    def _assert_path_within_root(root_dir: Path, target_path: Path) -> None:
        root_resolved = root_dir.resolve()
        target_resolved = target_path.resolve()
        try:
            target_resolved.relative_to(root_resolved)
        except ValueError as e:
            raise CompoxValidationError(
                f"Path traversal attempt detected: {target_path}",
                code="algorithm_export_security_error",
                cause=e,
            ) from e

    def _make_zip_from_directory(
        self, root_dir: Path, target_zip_path: str
    ) -> None:
        """
        Zip the entire directory tree under root_dir into target_zip_path.
        """
        target_zip_path = os.path.abspath(target_zip_path)
        with zipfile.ZipFile(target_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder, _, files in os.walk(root_dir):
                for filename in files:
                    file_path = Path(folder) / filename
                    # create archive name relative to root_dir
                    arcname = file_path.relative_to(root_dir)
                    zf.write(file_path, arcname=str(arcname))
