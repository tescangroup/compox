"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import uuid
import json
from datetime import datetime, timezone
import os
import shutil
import glob
import tempfile
import zipfile
import sys
import hashlib
import pathlib
import subprocess
import py_compile
import python_minifier
import toml
import warnings
from loguru import logger
import time

from compox.algorithm_utils.AlgorithmConfigSchema import (
    AlgorithmConfigSchema,
)
from compox.algorithm_utils.AlgorithmRecordRegistrar import (
    AlgorithmRecordRegistrar,
)
from compox.algorithm_utils.AlgorithmStorageMetrics import (
    AlgorithmStorageMetrics,
)
from compox.algorithm_utils.import_relativizer import (
    relativize_intra_package_imports,
)
from compox.database_connection import BaseConnection
from compox.exceptions import (
    CompoxAssetError,
    CompoxDeploymentError,
    CompoxError,
    CompoxImportError,
    CompoxValidationError,
)

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None


class AlgorithmDeployer:
    """
    The AlgorithmDeployer class is used to deploy an algorithm to the algorithm
    store in the database. The algorithm directory gets automatically separated
    into the algorithm module by detecting all .py files and zipping them as a
    python module and storing them in the module-store collection. Files other than
    .py files are stored as assets in the asset-store collection. The algorithm
    metadata is stored in the algorithm-store collection as a json file. The json file
    contains the algorithm name, major and minor version, the module id, the assets
    dictionary and the timestamp of when the algorithm was stored.

    Parameters
    ----------
    algorithm_directory : str
        The path to the algorithm directory.
    """

    _IGNORED_METADATA_DIRS = {".git", "__pycache__"}
    _IGNORED_METADATA_FILES = {".gitignore", ".gitmodules", ".gitattributes"}
    _MODULE_RENAME_RETRIES = 5
    _MODULE_RENAME_RETRY_DELAY_SECONDS = 0.5

    def __init__(
        self,
        algorithm_directory: str,
    ):

        self.logger = logger.bind(log_type="DEPLOYER")
        self._record_registrar = AlgorithmRecordRegistrar()
        self.algorithm_directory = algorithm_directory

        pyproject_toml = self.parse_pyproject_toml(algorithm_directory)
        self.algorithm_name = pyproject_toml["project"]["name"]
        self.algorithm_major_version = pyproject_toml["project"][
            "version"
        ].split(".")[0]
        self.algorithm_minor_version = pyproject_toml["project"][
            "version"
        ].split(".")[1]

        # check if ["tool"]["compox"] exists in pyproject.toml
        if "tool" in pyproject_toml and "compox" in pyproject_toml["tool"]:
            algorithm_config = pyproject_toml["tool"]["compox"]
            algorithm_config_schema = AlgorithmConfigSchema(**algorithm_config)
            # conver to dict to get the values
            algorithm_config_schema = algorithm_config_schema.model_dump()

            self.algorithm_type = algorithm_config_schema["algorithm_type"]
            self.tags = algorithm_config_schema["tags"]
            self.description = algorithm_config_schema["description"]
            self.device = algorithm_config_schema["supported_devices"]
            self.default_device = algorithm_config_schema["default_device"]
            self.additional_parameters = algorithm_config_schema[
                "additional_parameters"
            ]
            self.training_parameters = algorithm_config_schema[
                "training_parameters"
            ]
            self.benchmark_outputs = algorithm_config_schema[
                "benchmark_outputs"
            ]
            self.removable = algorithm_config_schema.get("removable", False)
            self.exportable = algorithm_config_schema.get("exportable", True)
        else:
            warnings.warn(
                (
                    "The [tool.compox] section is not found in pyproject.toml."
                    "Setting the algorithm type to Unspecified, tags to an empty list,"
                    "description to an empty string and additional parameters to an empty"
                    "list. If you want to specify the algorithm type, tags, description and"
                    "additional parameters, add the [tool.compox] section to the"
                    "algorithms's pyproject.toml."
                )
            )
            algorithm_config_schema = AlgorithmConfigSchema()
            # conver to dict to get the values
            algorithm_config_schema = algorithm_config_schema.model_dump()
            self.algorithm_type = algorithm_config_schema["algorithm_type"]
            self.tags = algorithm_config_schema["tags"]
            self.description = algorithm_config_schema["description"]
            self.device = algorithm_config_schema["supported_devices"]
            self.default_device = algorithm_config_schema["default_device"]
            self.additional_parameters = algorithm_config_schema[
                "additional_parameters"
            ]
            self.training_parameters = algorithm_config_schema[
                "training_parameters"
            ]
            self.benchmark_outputs = algorithm_config_schema[
                "benchmark_outputs"
            ]
            self.removable = algorithm_config_schema.get("removable", False)
            self.exportable = algorithm_config_schema.get("exportable", True)

        self.check_importable = (
            pyproject_toml.get("tool", {})
            .get("compox", {})
            .get("check_importable", False)
        )
        self.obfuscate = (
            pyproject_toml.get("tool", {})
            .get("compox", {})
            .get("obfuscate", True)
        )
        self.pyc_only = (
            pyproject_toml.get("tool", {})
            .get("compox", {})
            .get("pyc_only", True)
        )
        tool_compox = pyproject_toml.get("tool", {}).get("compox", {})
        if "hash_module" in tool_compox or "hash_assets" in tool_compox:
            warnings.warn(
                "hash_module/hash_assets are deprecated and ignored. "
                "Deduplication by content hash is always enabled.",
                DeprecationWarning,
            )
        self.hash_module = tool_compox.get("hash_module", True)
        self.hash_assets = tool_compox.get("hash_assets", True)

    @staticmethod
    def _find_algorithm_root(extracted_root: str) -> str:
        """
        Find the algorithm root folder within an extracted zip directory.
        If a single top-level directory exists and contains pyproject.toml,
        return that directory, otherwise return the extracted root.
        """
        entries = [
            os.path.join(extracted_root, name)
            for name in os.listdir(extracted_root)
        ]
        directories = [p for p in entries if os.path.isdir(p)]
        if len(directories) == 1 and os.path.isfile(
            os.path.join(directories[0], "pyproject.toml")
        ):
            return directories[0]
        return extracted_root

    @classmethod
    def deploy_from_zip(
        cls,
        zip_path: str,
        database_connection: BaseConnection.BaseConnection | None = None,
        algorithm_name_override: str | None = None,
        algorithm_major_version_override: str | None = None,
        algorithm_collection_name: str = "algorithm-store",
        module_collection_name: str = "module-store",
        asset_collection_name: str = "asset-store",
    ) -> str:
        """
        Deploy an algorithm from a zip archive containing the algorithm files.

        Parameters
        ----------
        zip_path : str
            Path to the algorithm zip archive.
        database_connection : BaseConnection.BaseConnection | None
            The database connection object.
        algorithm_name_override : str | None
            The algorithm name override.
        algorithm_major_version_override : str | None
            The algorithm major version override.
        algorithm_collection_name : str, optional
            The name of the collection to store the algorithm.
        module_collection_name : str, optional
            The name of the collection to store the module.
        asset_collection_name : str, optional
            The name of the collection to store the assets.

        Returns
        -------
        str
            algorithm id
        """
        if not os.path.isfile(zip_path):
            raise CompoxValidationError(
                f"Algorithm deployment zip not found: {zip_path}",
                code="deployment_zip_not_found",
                details={"path": zip_path},
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp_dir)
            except zipfile.BadZipFile as e:
                raise CompoxValidationError(
                    f"Invalid algorithm deployment zip: {zip_path}",
                    code="invalid_deployment_zip",
                    details={"path": zip_path},
                    cause=e,
                ) from e

            try:
                algorithm_root = cls._find_algorithm_root(tmp_dir)
                deployer = cls(algorithm_root)
                return deployer.store_algorithm(
                    database_connection=database_connection,
                    algorithm_name_override=algorithm_name_override,
                    algorithm_major_version_override=algorithm_major_version_override,
                    algorithm_collection_name=algorithm_collection_name,
                    module_collection_name=module_collection_name,
                    asset_collection_name=asset_collection_name,
                )
            except CompoxError:
                raise
            except Exception as e:
                raise CompoxDeploymentError(
                    "Failed to deploy algorithm from zip archive",
                    code="deployment_zip_failed",
                    details={"path": zip_path},
                    cause=e,
                ) from e

    def parse_pyproject_toml(self, path_to_algorithm_directory: str) -> dict:
        """
        Parse the pyproject.toml file in the algorithm directory to get the
        algorithm name, major version and minor version.

        Parameters
        ----------
        path_to_algorithm_directory : str
            The path to the algorithm directory.

        Returns
        -------
        dict
            The algorithm name, major version and minor version.

        Raises
        ------
        FileNotFoundError
            If pyproject.toml not found in algorithm directory.

        """
        pyproject_path = os.path.join(
            path_to_algorithm_directory, "pyproject.toml"
        )
        if not os.path.exists(pyproject_path):
            raise FileNotFoundError(
                f"pyproject.toml file not found in {path_to_algorithm_directory}."
            )

        if tomllib is not None:
            try:
                with open(pyproject_path, "rb") as f:
                    pyproject_toml = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise ValueError(
                    f"Invalid pyproject.toml in {path_to_algorithm_directory}: {e}."
                ) from e
        else:
            try:
                with open(pyproject_path, encoding="utf-8") as f:
                    pyproject_toml = toml.load(f)
            except toml.TomlDecodeError as e:
                raise ValueError(
                    f"Invalid pyproject.toml in {path_to_algorithm_directory}: {e}."
                ) from e

        return pyproject_toml

    def store_algorithm(
        self,
        database_connection: BaseConnection.BaseConnection | None = None,
        algorithm_name_override: str | None = None,
        algorithm_major_version_override: str | None = None,
        algorithm_collection_name: str = "algorithm-store",
        module_collection_name: str = "module-store",
        asset_collection_name: str = "asset-store",
    ) -> str:
        """

        Store the algorithm to the algorithm store.

        Parameters
        ----------
        database_connection : BaseConnection.BaseConnection | None
            The database connection object. Can be None if the algorithm is not
            supposed to be stored in the database (e.g. for local testing and
            development).
        algorithm_name_override : str | None
            The algorithm name override.
        algorithm_major_version_override : str | None
            The algorithm major version override.
        algorithm_collection_name : str, optional
            The name of the collection to store the algorithm. The default is
            "algorithm-store".
        module_collection_name : str, optional
            The name of the collection to store the module. The default is
            "module-store".
        asset_collection_name : str, optional
            The name of the collection to store the assets. The default is
            "asset-store".

        Returns
        -------
        str
            algorithm id

        Raises
        ------
        Exception
            if algorithm module or assets store failed
        """

        if algorithm_name_override is not None:
            self.algorithm_name = algorithm_name_override
        if algorithm_major_version_override is not None:
            self.algorithm_major_version = algorithm_major_version_override
        existing_algorithm_record = None
        # check if the algorithm already exists in the database
        if database_connection is not None:
            existing_algorithm_record = self._record_registrar.find_existing_algorithm_by_name_and_major(
                database_connection,
                self.algorithm_name,
                self.algorithm_major_version,
            )
            if existing_algorithm_record is not None:
                self.logger.info(
                    f"""Algorithm {self.algorithm_name} with major version {self.algorithm_major_version} 
                    already exists in the database with id {existing_algorithm_record[1]['algorithm_id']}. Modifying the existing algorithm."""
                )
                algorithm_id = existing_algorithm_record[1]["algorithm_id"]

            else:
                algorithm_id = self.generate_uuid()
        else:
            existing_algorithm_record = None
            algorithm_id = self.generate_uuid()

        # store the algorithm module
        try:
            (
                algorithm_module_id,
                algorithm_module_bytes,
            ) = self._create_algorithm_module(
                self.algorithm_directory,
                check_importable=self.check_importable,
                obfuscate=self.obfuscate,
                pyc_only=self.pyc_only,
            )
            self.logger.info(
                f"Created algorithm module with id: {algorithm_module_id}"
            )
        except CompoxError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to create algorithm module: {e}")
            raise CompoxImportError(
                "Failed to create algorithm module",
                code="algorithm_module_create_failed",
                cause=e,
            ) from e

        if database_connection is not None:
            try:
                if (
                    module_collection_name
                    not in database_connection.list_collections()
                ):
                    database_connection.create_collections(
                        [module_collection_name]
                    )

                # check if module alread exists in the storage
                module_ids_in_storage = database_connection.list_objects(
                    module_collection_name
                )
                module_id_in_storage = [
                    m["Key"] == algorithm_module_id
                    for m in module_ids_in_storage
                ]

                if any(module_id_in_storage):
                    self.logger.info(
                        f"Algorithm module with id: {algorithm_module_id} already exists in the storage. Skipping upload."
                    )
                else:
                    database_connection.put_objects(
                        module_collection_name,
                        [algorithm_module_id],
                        [algorithm_module_bytes],
                    )
                    self.logger.info(
                        f"Stored algorithm module with id: {algorithm_module_id}"
                    )
            except CompoxError:
                raise
            except Exception as e:
                self.logger.error(f"Failed to store algorithm module: {e}")
                raise CompoxDeploymentError(
                    "Failed to store algorithm module",
                    code="algorithm_module_store_failed",
                    cause=e,
                ) from e

        # store the algorithm assets
        try:
            algorithm_assets_dict = self._store_algorithm_assets(
                self.algorithm_directory,
                database_connection=database_connection,
                collection_name=asset_collection_name,
            )
            self.logger.info(
                f"Stored algorithm assets: {algorithm_assets_dict}"
            )
        except CompoxError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to store algorithm assets: {e}")
            raise CompoxAssetError(
                "Failed to store algorithm assets",
                code="algorithm_assets_store_failed",
                cause=e,
            ) from e

        # get the timestamp
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # compose the algorithm json

        if existing_algorithm_record is None:
            algorithm_json = (
                self._record_registrar.compose_new_algorithm_record(
                    algorithm_id=algorithm_id,
                    algorithm_name=self.algorithm_name,
                    algorithm_major_version=self.algorithm_major_version,
                    algorithm_minor_version=self.algorithm_minor_version,
                    module_id=algorithm_module_id,
                    assets_dict=algorithm_assets_dict,
                    metadata={
                        "algorithm_type": self.algorithm_type,
                        "algorithm_tags": self.tags,
                        "algorithm_description": self.description,
                        "supported_devices": self.device,
                        "default_device": self.default_device,
                        "additional_parameters": self.additional_parameters,
                        "training_parameters": self.training_parameters,
                        "benchmark_outputs": self.benchmark_outputs,
                        "removable": self.removable,
                        "exportable": self.exportable,
                    },
                    timestamp=timestamp,
                )
            )
            record_modified = True
        else:
            algorithm_json, record_modified = (
                self._insert_new_minor_version_to_existing_algorithm(
                    existing_algorithm_record[1],
                    algorithm_module_id,
                    algorithm_assets_dict,
                )
            )
            algorithm_json["exportable"] = self.exportable

        # store the algorithm json in the algorithm-store collection
        # check if the collection exists and create it if it does not

        if database_connection is not None:
            algorithm_json = AlgorithmStorageMetrics(
                database_connection
            ).attach_metrics(algorithm_json)
            self._record_registrar.upsert_algorithm_record(
                database_connection=database_connection,
                algorithm_record=algorithm_json,
                algorithm_collection_name=algorithm_collection_name,
                existing_algorithm=existing_algorithm_record,
            )
        self.logger.info(
            f"Stored algorithm json: {json.dumps(algorithm_json, indent=4)}"
        )
        return algorithm_id

    def _insert_new_minor_version_to_existing_algorithm(
        self,
        existing_algorithm_record: dict,
        module_id: str,
        assets_dict: dict,
    ) -> tuple[dict, bool]:
        """
        Insert a new minor version using the shared algorithm record registrar.

        Parameters
        ----------
        existing_algorithm_record : dict
            The existing algorithm record.
        module_id : str
            The module id.
        assets_dict : dict
            The assets dictionary.
        Returns
        -------
        tuple[dict, bool]
            The modified algorithm record and a boolean indicating if a new minor
            version was inserted.
        """
        modified_algorithm_json, record_modified = (
            self._record_registrar.insert_new_minor_version(
                existing_algorithm_record=existing_algorithm_record,
                module_id=module_id,
                assets_dict=assets_dict,
            )
        )
        if not record_modified:
            self.logger.info(
                "The module id and assets dictionary are the same as the latest minor version. Not inserting a new minor version."
            )
        return modified_algorithm_json, record_modified

    def _create_algorithm_module(
        self,
        path_to_algorithm_directory: str,
        check_importable: bool = False,
        obfuscate: bool = False,
        pyc_only: bool = False,
    ) -> tuple[str, bytes]:
        """
        Detects all .py files in the algorithm directory, zips them as a python
        module and stores them in the module collection.

        Parameters
        ----------
        path_to_algorithm_directory : str
            The path to the algorithm directory.

        database_connection : BaseConnection.BaseConnection | None
            The database connection object.

        check_importable : bool, optional
            Whether to check if the module is importable by performing an import
            test. The default is False.

        obfuscate : bool, optional
            Whether to obfuscate the .py files. The default is False.

        Returns
        -------
        str
            The module id.

        bytes
            The module bytes.

        Raises
        ------
        CompoxValidationError
            if Runner.py / Runner.pyc is missing.
        CompoxImportError
            if importability check fails.
        """

        # Source algorithms are transformed from .py files. Exported/runtime
        # artifacts may instead provide a bytecode-only package with Runner.pyc.
        py_files, _ = self.find_py_files(path_to_algorithm_directory)
        pyc_files, _ = self.find_pyc_files(path_to_algorithm_directory)
        py_files_with_relative_path = [
            os.path.relpath(py_file, path_to_algorithm_directory)
            for py_file in py_files
        ]
        pyc_files_with_relative_path = [
            os.path.relpath(pyc_file, path_to_algorithm_directory)
            for pyc_file in pyc_files
        ]

        has_runner_py = "Runner.py" in py_files_with_relative_path
        has_runner_pyc = "Runner.pyc" in pyc_files_with_relative_path

        if not has_runner_py and not has_runner_pyc:
            raise CompoxValidationError(
                "Runner.py / Runner.pyc not found in the root of the algorithm directory.",
                code="algorithm_runner_not_found",
                details={"algorithm_directory": path_to_algorithm_directory},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Build under a stable staging package name first. We compute module_id
            # only after all code-transforming steps are applied.
            staging_package_name = "__module__"
            root_module_dir = os.path.join(temp_dir, "module_build")
            module_path = os.path.join(root_module_dir, staging_package_name)
            os.makedirs(module_path, exist_ok=True)
            # create __init__.py file in the root of the module directory
            init_file_path = os.path.join(root_module_dir, "__init__.py")
            with open(init_file_path, "w", encoding="utf-8") as f:
                f.write("# Init file for the module\n")

            if has_runner_py:
                # copy the .py files to the temporary directory while preserving the directory structure
                for py_file, py_file_with_relative_path in zip(
                    py_files, py_files_with_relative_path
                ):
                    os.makedirs(
                        os.path.join(
                            module_path,
                            os.path.dirname(py_file_with_relative_path),
                        ),
                        exist_ok=True,
                    )
                    shutil.copy(
                        py_file,
                        os.path.join(module_path, py_file_with_relative_path),
                    )

                relativize_intra_package_imports(module_path)

                module_py_files, _ = self.find_py_files(module_path)

                # check if __init__.py exists in the module_path
                if not os.path.exists(os.path.join(module_path, "__init__.py")):
                    # create an empty __init__.py file
                    with open(
                        os.path.join(module_path, "__init__.py"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write("# Init file for the package\n")

                # load the content of the __init__.py file and append a runner import statement
                with open(
                    os.path.join(module_path, "__init__.py"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    init_content = f.read()
                init_content += "\nfrom .Runner import Runner\n"
                with open(
                    os.path.join(module_path, "__init__.py"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(init_content)

                if obfuscate:
                    # obfuscate the .py files
                    # TODO: implement obfuscation
                    # for now, minimize the .py files
                    self._minimize_py_files(module_py_files)

                # Optionally compile to bytecode-only package.
                # Keep legacy .pyc file layout next to module paths for predictable
                # loading from extracted archives.
                if pyc_only:
                    module_pyc_files = self._compile_and_strip_py_files(
                        module_py_files, module_root=module_path
                    )
                    files_for_hash = module_pyc_files
                else:
                    files_for_hash = module_py_files
            else:
                # Compiled artifact input: preserve .pyc package layout as-is.
                for pyc_file, pyc_rel_path in zip(
                    pyc_files, pyc_files_with_relative_path
                ):
                    os.makedirs(
                        os.path.join(
                            module_path,
                            os.path.dirname(pyc_rel_path),
                        ),
                        exist_ok=True,
                    )
                    shutil.copy(
                        pyc_file,
                        os.path.join(module_path, pyc_rel_path),
                    )
                files_for_hash = self.find_pyc_files(module_path)[0]

            # Compute module id from transformed sources to ensure packaging options
            # (e.g., obfuscation) affect deduplication.
            module_id = self.get_py_files_hashes(
                files_for_hash, base_directory=module_path
            )

            final_module_path = os.path.join(root_module_dir, module_id)
            self._rename_module_path_with_retries(
                module_path, final_module_path
            )

            # create a temporary zip file of the temporary directory
            shutil.make_archive(root_module_dir, "zip", root_module_dir)

            if check_importable:
                importable = self.check_if_zip_is_importable(
                    root_module_dir + ".zip", module_id
                )
                if not importable:
                    raise CompoxImportError(
                        "The current environment cannot import the algorithm module. Check the above logs for more details about the ImportError cause. This check can be disabled by setting check_importable to False in the affected algorithm's pyproject.toml file.",
                        code="algorithm_module_import_check_failed",
                        details={"module_id": module_id},
                    )

            with open(root_module_dir + ".zip", "rb") as f:
                module_bytes = f.read()

        return module_id, module_bytes

    def _rename_module_path_with_retries(
        self,
        source_path: str,
        target_path: str,
        retries: int | None = None,
        delay_seconds: float | None = None,
    ) -> None:
        """
        Rename the staged module directory, tolerating short transient file locks.

        On Windows, antivirus/indexing tools can briefly lock newly written
        Python files. In that case ``os.rename`` can fail even though retrying a
        moment later succeeds.
        """
        retry_count = retries or self._MODULE_RENAME_RETRIES
        retry_delay = (
            self._MODULE_RENAME_RETRY_DELAY_SECONDS
            if delay_seconds is None
            else delay_seconds
        )

        for attempt in range(1, retry_count + 1):
            try:
                os.rename(source_path, target_path)
                return
            except PermissionError as e:
                if attempt >= retry_count:
                    raise
                self.logger.warning(
                    "Cannot rename module path, retrying "
                    f"({attempt}/{retry_count}): {e}"
                )
                time.sleep(retry_delay)

    @staticmethod
    def _compile_and_strip_py_files(
        py_files: list[str], module_root: str
    ) -> list[str]:
        """
        Compile Python sources to adjacent ``.pyc`` files and remove sources.

        Parameters
        ----------
        py_files : list[str]
            Absolute paths to source ``.py`` files.
        module_root : str
            Root module directory used to preserve relative layout.

        Returns
        -------
        list[str]
            Absolute paths to generated ``.pyc`` files.
        """
        pyc_files: list[str] = []
        for py_file in py_files:
            rel = os.path.relpath(py_file, module_root)
            rel_no_ext = os.path.splitext(rel)[0]
            pyc_path = os.path.join(module_root, rel_no_ext + ".pyc")
            os.makedirs(os.path.dirname(pyc_path), exist_ok=True)
            py_compile.compile(
                py_file,
                cfile=pyc_path,
                dfile=pathlib.Path(rel).as_posix(),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
            )
            pyc_files.append(pyc_path)

        for py_file in py_files:
            os.remove(py_file)

        return pyc_files

    def _store_algorithm_assets(
        self,
        path_to_algorithm_directory: str,
        database_connection: BaseConnection.BaseConnection | None = None,
        collection_name: str = "asset-store",
    ) -> dict:
        """
        Stores the assets of the algorithm in the asset-store collection.

        Parameters
        ----------
        path_to_algorithm_directory : str
            The path to the algorithm directory.

        database_connection : BaseConnection.BaseConnection | None
            The database connection object.

        collection_name : str, optional
            The name of the collection to store the assets. The default is
            "asset-store".

        Returns
        -------
        dict
            The dictionary of the assets.

        """

        # get all the files other than .py files in the algorithm directory
        other_than_py_files = self.find_other_than_py_files(
            path_to_algorithm_directory
        )
        other_than_py_files_with_relative_path = [
            os.path.relpath(file, path_to_algorithm_directory)
            for file in other_than_py_files
        ]

        # store the assets in the asset-store collection
        assets_dict = {}
        for file, relative_file in zip(
            other_than_py_files, other_than_py_files_with_relative_path
        ):
            with open(file, "rb") as f:
                file_bytes = f.read()

            if database_connection is not None:
                if (
                    collection_name
                    not in database_connection.list_collections()
                ):
                    database_connection.create_collections([collection_name])

                file_hash = hashlib.md5(file_bytes).hexdigest()

                # check if the file already exists
                file_exists = database_connection.check_objects_exist(
                    collection_name, [file_hash]
                )[0]

                if not file_exists:
                    database_connection.put_objects(
                        collection_name,
                        [file_hash],
                        [file_bytes],
                    )
                    self.logger.info(
                        f"Uploaded new asset {relative_file} to {collection_name}"
                    )
                else:
                    self.logger.info(
                        f"Asset {relative_file} already exists in {collection_name}. Reusing existing asset."
                    )
                assets_dict[self.process_path_to_dict_key(relative_file)] = (
                    file_hash
                )
        return assets_dict

    @staticmethod
    def process_path_to_dict_key(path: str) -> str:
        """
        This method takes a path to a file specified as directory and substitutes
        any backslashes and double backslashes with forward slashes. It also removes
        the leading forward slash if it exists. This is necessary to store the
        directory structure as a dictionary key which can then be accessed by the
        server independently of the operating system, where the deployment is
        performed.

        Parameters
        ----------
        path : str
            A path to a file.

        Returns
        -------
        str
            The processed path with forward slashes and without the leading forward
            slash.

        """
        return path.lstrip("/\\").replace("\\", "/")

    def _minimize_py_files(self, py_files: list[str]) -> None:
        """
        Minimize .py files using python_minifier. The minification is performed
        in place.

        Parameters
        ----------
        py_files : list[str]
            The list of paths to the .py files to minimize.

        Returns
        -------
        None

        """
        for py_file in py_files:
            with open(py_file, "r", encoding="utf-8") as f:
                file_content = f.read()
            minified_content = python_minifier.minify(
                file_content, remove_literal_statements=True
            )
            with open(py_file, "w", encoding="utf-8") as f:
                f.write(minified_content)

    @staticmethod
    def check_if_zip_is_importable(path_to_zip: str, module_name: str) -> bool:
        """
        Check if the zipped compox module is importable. This serves as a sanity
        check that the environment where the algorithm is being deployed has the
        necessary dependencies available.

        Parameters
        ----------
        path_to_zip : str
            The path to the zip file.
        module_name : str
            The name of the module.

        Returns
        -------
        bool
            True if the module is importable, False otherwise.

        """
        check_script = (
            "import importlib\n"
            "import sys\n"
            "\n"
            "path_to_zip = sys.argv[1]\n"
            "module_name = sys.argv[2]\n"
            "\n"
            "sys.path.insert(0, path_to_zip)\n"
            "module = importlib.import_module(module_name)\n"
            "runner = getattr(module, 'Runner', None)\n"
            "if runner is None:\n"
            "    raise AttributeError('Runner class not found in module')\n"
            "runner()\n"
        )

        try:
            result = subprocess.run(
                [sys.executable, "-c", check_script, path_to_zip, module_name],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except Exception as e:
            logger.error(
                f"ImportError: failed to run import check subprocess: {e}"
            )
            return False

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            msg = stderr if stderr else stdout
            logger.error(f"ImportError: {msg}")
            return False

        return True

    @staticmethod
    def calculate_etag(file: bytes) -> str:
        """
        Calculate the etag hash of a file.

        Parameters
        ----------
        file : bytes
            The file.

        Returns
        -------
        str
            The etag hash.

        """
        import hashlib

        md5s = hashlib.md5(file)
        return '"{}"'.format(md5s.hexdigest())

    @staticmethod
    def find_py_files(
        directory: str, ignore_pycache: bool = True
    ) -> tuple[list[str], str]:
        """
        Find all the .py files in a directory recursively.

        Parameters
        ----------
        directory : str
            The directory to search.

        ignore_pycache: bool
            Whether to ignore __pycache__ directory

        Returns
        -------
        tuple[list[str], str]
            A tuple containing a list of py files in a directory and a string
            representing their combined hash.

        """
        py_files = []

        ignored_dirs = {".git"}
        if ignore_pycache:
            ignored_dirs |= AlgorithmDeployer._IGNORED_METADATA_DIRS - {".git"}

        for root, dirs, _ in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            py_files.extend(glob.glob(os.path.join(root, "*.py")))

        files_hash = AlgorithmDeployer.get_py_files_hashes(
            py_files, base_directory=directory
        )
        return py_files, files_hash

    @staticmethod
    def find_pyc_files(
        directory: str, ignore_pycache: bool = True
    ) -> tuple[list[str], str]:
        """
        Find all the .pyc files in a directory recursively.

        Parameters
        ----------
        directory : str
            The directory to search.
        ignore_pycache: bool
            Whether to ignore __pycache__ directory.

        Returns
        -------
        tuple[list[str], str]
            A tuple containing a list of .pyc files and their combined hash.
        """
        pyc_files = []

        for root, _, _ in os.walk(directory):
            pyc_files.extend(glob.glob(os.path.join(root, "*.pyc")))

        if ignore_pycache:
            pyc_files = [
                file
                for file in pyc_files
                if "__pycache__" not in file.split(os.path.sep)
            ]

        files_hash = AlgorithmDeployer.get_py_files_hashes(
            pyc_files, base_directory=directory
        )
        return pyc_files, files_hash

    @staticmethod
    def get_py_files_hashes(
        py_files: list[str], base_directory: str | None = None
    ) -> str:
        """
        Get a combined hash of all the .py files.
        Parameters
        ----------
        py_files : list[str]
            The list of .py files.
        Returns
        -------
        str
            The combined md5 hash of all the .py files.
        """
        combined_md5 = hashlib.md5()

        files_with_rel_paths: list[tuple[str, str]] = []
        for py_file in py_files:
            if base_directory is not None:
                rel_path = os.path.relpath(py_file, base_directory)
            else:
                rel_path = os.path.basename(py_file)
            # Normalize separators to make hash stable across platforms.
            normalized_rel_path = pathlib.Path(rel_path).as_posix()
            files_with_rel_paths.append((normalized_rel_path, py_file))

        for normalized_rel_path, py_file in sorted(files_with_rel_paths):
            combined_md5.update(normalized_rel_path.encode("utf-8"))
            combined_md5.update(b"\x00")
            with open(py_file, "rb") as f:
                while True:
                    data = f.read(65536)
                    if not data:
                        break
                    combined_md5.update(data)
        return combined_md5.hexdigest()

    @staticmethod
    def find_other_than_py_files(
        directory: str,
        ignore_pycache: bool = True,
        ignore_gitignore: bool = True,
    ) -> list[str]:
        """
        Find all the files in a directory other than .py files.

        Parameters
        ----------
        directory : str
            The directory to search.

        ignore_pycache : bool, optional
            Whether to ignore the __pycache__ directory. The default is True.

        ignore_gitignore : bool, optional
            Whether to ignore the .gitignore file. The default is True.

        Returns
        -------
        list[str]
            The list of files other than .py files.

        """
        other_than_py_files = []

        ignored_dirs = {".git"}
        if ignore_pycache:
            ignored_dirs |= AlgorithmDeployer._IGNORED_METADATA_DIRS - {".git"}

        # get all the files other than .py files in the algorithm directory
        for root, dirs, _ in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            other_than_py_files.extend(glob.glob(os.path.join(root, "*")))

        # remove Python module files from the list
        other_than_py_files = [
            file
            for file in other_than_py_files
            if not file.endswith(".py") and not file.endswith(".pyc")
        ]
        # remove paths that are directories
        other_than_py_files = [
            file for file in other_than_py_files if not os.path.isdir(file)
        ]

        if ignore_gitignore:
            other_than_py_files = [
                file
                for file in other_than_py_files
                if not (
                    set(file.split(os.path.sep))
                    & AlgorithmDeployer._IGNORED_METADATA_FILES
                )
            ]

        return other_than_py_files

    @staticmethod
    def generate_uuid(version: int = 1) -> str:
        """
        Generate a uuid.

        Parameters
        ----------
        version : int, optional
            The version of the uuid. The default is 1.

        Returns
        -------
        str
            The uuid.

        Raises
        ------
        ValueError
            if version of the uuid is not 1 or 4.
        """
        if version == 1:
            return str(uuid.uuid1())
        elif version == 4:
            return str(uuid.uuid4())
        else:
            raise ValueError("uuid version must be 1 or 4")


if __name__ == "__main__":

    algorithm_deployer = AlgorithmDeployer(
        "C:\\Users\\Jan Matula\\Work\\python-computing-backend\\compox\\algorithms\\test_training_runner"
    )
    algorithm_deployer.store_algorithm(database_connection=None)
