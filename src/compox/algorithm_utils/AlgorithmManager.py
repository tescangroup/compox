"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import json
from loguru import logger

from compox.database_connection import BaseConnection
from compox.algorithm_utils.AlgorithmStorageMetrics import (
    AlgorithmStorageMetrics,
)
from compox.training.AlgorithmCheckpoint import AlgorithmCheckpoint


class AlgorithmManager:
    """
    This class is responsible for managing the algorithms, modules and assets in
    the database. It provides methods to list or delete algorithms, modules and assets.
    To store the algorithms, modules and assets, use the AlgorithmDeployer class.

    Parameters
    ----------
    database_connection : BaseConnection.BaseConnection
        The database connection to use for the operations.

    algorithms_collection : str
        The name of the collection where the algorithms are stored.

    module_collection: str
        The name of the collection where the modules are stored.

    assets_collection : str
        The name of the collection where the assets are stored.
    """

    def __init__(
        self,
        database_connection: BaseConnection.BaseConnection,
        algorithms_collection: str = "algorithm-store",
        module_collection: str = "module-store",
        assets_collection: str = "asset-store",
        checkpoint_collection: str = "algorithm-checkpoint-store",
    ):
        self.database_connection = database_connection
        self.algorithms_collection = algorithms_collection
        self.module_collection = module_collection
        self.assets_collection = assets_collection
        self.checkpoint_collection = checkpoint_collection
        self.logger = logger.bind(
            log_type="ALGORITHM MANAGER",
        )

    def list_algorithms(
        self,
        name: str | None = None,
        major_version: str | None = None,
    ) -> list[dict]:
        """
        List all algorithms stored in the database. Optionally can filter by name or
        major version of the algorithm.

        Parameters
        ----------
        name : str | None, optional
            Can be used to filter the algorithms by name.

        major_version : str | None, optional
            Can be used to filter the algorithms by major version.

        Returns
        -------
        list[dict]
            The list of algorithms defined by their jsons

        """
        algorithms = self.database_connection.list_objects(
            self.algorithms_collection
        )

        if len(algorithms) == 0:
            return []

        algorithms_jsons = []

        # get the jsons
        for algorithm in algorithms:
            algorithm_json = self.database_connection.get_objects(
                self.algorithms_collection, [algorithm["Key"]]
            )
            algorithms_jsons.append(dict(json.loads(algorithm_json[0])))

        # filter the algorithms
        if name:
            algorithms_jsons = [
                algorithm
                for algorithm in algorithms_jsons
                if name == algorithm["algorithm_name"]
            ]

        if major_version:
            algorithms_jsons = [
                algorithm
                for algorithm in algorithms_jsons
                if major_version == algorithm["algorithm_major_version"]
            ]

        return algorithms_jsons

    def delete_algorithm(
        self,
        name: str | None = None,
        major_version: str | None = None,
    ) -> None:
        """
        Delete an algorithm and associated modules and assets.

        Parameters
        ----------
        name : str | None
            The name of the algorithm to delete.

        major_version : str | None
            The major version of the algorithm to delete.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            if name or major_version is not specified
        """

        if not name or not major_version:
            raise ValueError(
                "You must specify the name and major_version of the algorithm to delete."
            )

        algorithms = self.list_algorithms(name, major_version)

        for algorithm in algorithms:

            checkpoints = algorithm.get("checkpoints", [])
            minor_version = algorithm.get("algorithm_minor_version", {})

            algorithm_id = algorithm["algorithm_id"]
            algorithm_name = algorithm["algorithm_name"]
            algorithm_major_version = algorithm["algorithm_major_version"]
            self.logger.info(
                f"Deleting algorithm {algorithm_name} "
                f"{algorithm_major_version}."
            )

            algorithm_key = (
                f"{algorithm_id}~{algorithm_name}~{algorithm_major_version}"
            )
            algorithm_module_pairs = self._get_all_algorithm_module_pairs()
            algorithm_asset_pairs = self._get_all_algorithm_asset_pairs()
            # go through the minor versions
            for minor_ver in minor_version.values():
                # delete the modules
                self._delete_module(
                    current_algorithm_key=algorithm_key,
                    module_id=minor_ver["module_id"],
                    algorithm_module_pairs=algorithm_module_pairs,
                )
                # delete the assets
                assets = minor_ver.get("assets", {})
                for key, value in assets.items():
                    self._delete_asset(
                        current_algorithm_key=algorithm_key,
                        asset_id=value,
                        algorithm_asset_pairs=algorithm_asset_pairs,
                    )
            # delete the checkpoints
            for checkpoint_id in checkpoints:
                checkpoint = AlgorithmCheckpoint(
                    self.database_connection, checkpoint_id=checkpoint_id
                )
                checkpoint.delete_checkpoint()

            # delete the algorithm
            self.database_connection.delete_objects(
                self.algorithms_collection, [algorithm_key]
            )

        return None

    def delete_algorithm_minor_version(
        self,
        name: str,
        major_version: str,
        minor_version: str,
    ) -> None:
        """
        Delete a specific minor version of an algorithm. The algorithm itself is
        not deleted, as long as there are other minor versions present. If the
        last minor version is deleted, the entire algorithm is deleted.

        Delete associated modules and assets if they are not used by other algorithms/minor versions.

        Parameters
        ----------
        name : str
            The name of the algorithm.
        major_version : str
            The major version of the algorithm.
        minor_version : str
            The minor version of the algorithm to delete.
        Returns
        -------
        None
        """
        algorithms = self.list_algorithms(name, major_version)

        for algorithm in algorithms:
            algorithm_id = algorithm["algorithm_id"]
            algorithm_name = algorithm["algorithm_name"]
            algorithm_major_version = algorithm["algorithm_major_version"]
            self.logger.info(
                f"Deleting minor version {minor_version} of algorithm "
                f"{algorithm_name} {algorithm_major_version}."
            )

            algorithm_key = (
                f"{algorithm_id}~{algorithm_name}~{algorithm_major_version}"
            )
            algorithm_module_pairs = self._get_all_algorithm_module_pairs()
            algorithm_asset_pairs = self._get_all_algorithm_asset_pairs()

            minor_versions = algorithm.get("algorithm_minor_version", {})

            if minor_version in minor_versions:
                minor_ver = minor_versions[minor_version]
                # delete the module
                self._delete_module(
                    current_algorithm_key=algorithm_key,
                    module_id=minor_ver["module_id"],
                    algorithm_module_pairs=algorithm_module_pairs,
                )
                # delete the assets
                assets = minor_ver.get("assets", {})
                for key, value in assets.items():
                    self._delete_asset(
                        current_algorithm_key=algorithm_key,
                        asset_id=value,
                        algorithm_asset_pairs=algorithm_asset_pairs,
                    )

                # remove the minor version from the algorithm record
                del minor_versions[minor_version]

                # update latest minor version if needed
                if algorithm["latest_algorithm_minor_version"] == minor_version:
                    if len(minor_versions) > 0:
                        latest_minor = max(
                            int(ver) for ver in minor_versions.keys()
                        )
                        algorithm["latest_algorithm_minor_version"] = str(
                            latest_minor
                        )
                    else:
                        algorithm["latest_algorithm_minor_version"] = None

                # update the algorithm record in the database
                algorithm = AlgorithmStorageMetrics(
                    self.database_connection
                ).attach_metrics(algorithm)
                self.database_connection.put_objects(
                    self.algorithms_collection,
                    [
                        f"{algorithm_id}~{algorithm_name}~{algorithm_major_version}"
                    ],
                    [json.dumps(algorithm)],
                )

                # if no minor versions left, delete the entire algorithm
                if len(minor_versions) == 0:
                    self.logger.info(
                        f"No minor versions left for algorithm "
                        f"{algorithm_name} {algorithm_major_version}. "
                        f"Deleting entire algorithm."
                    )
                    self.database_connection.delete_objects(
                        self.algorithms_collection,
                        [
                            f"{algorithm_id}~{algorithm_name}~{algorithm_major_version}"
                        ],
                    )
        return None

    def _get_all_algorithm_module_pairs(self) -> list[dict]:
        """
        Get all algorithms and their associated modules.

        Returns
        -------
        list[dict]
            A list of dictionaries containing the algorithm key and module id.
        """
        all_algorithms = self.list_algorithms()
        algorithm_module_pairs = []
        for algorithm in all_algorithms:
            algorithm_key = (
                f"{algorithm['algorithm_id']}~"
                f"{algorithm['algorithm_name']}~"
                f"{algorithm['algorithm_major_version']}"
            )
            minor_versions = algorithm.get("algorithm_minor_version", {})
            for minor_ver in minor_versions.values():
                module_id = minor_ver["module_id"]
                algorithm_module_pairs.append(
                    {
                        "algorithm_key": algorithm_key,
                        "module_id": module_id,
                    }
                )
        return algorithm_module_pairs

    def _get_all_algorithm_asset_pairs(self) -> list[dict]:
        """
        Get all algorithms and their associated assets.

        Returns
        -------
        list[dict]
            A list of dictionaries containing the algorithm key and asset ids.
        """
        all_algorithms = self.list_algorithms()
        algorithm_asset_pairs = []
        for algorithm in all_algorithms:
            algorithm_key = (
                f"{algorithm['algorithm_id']}~"
                f"{algorithm['algorithm_name']}~"
                f"{algorithm['algorithm_major_version']}"
            )
            minor_versions = algorithm.get("algorithm_minor_version", {})
            for minor_ver in minor_versions.values():
                assets = minor_ver.get("assets", {})
                for asset_id in assets.values():
                    algorithm_asset_pairs.append(
                        {
                            "algorithm_key": algorithm_key,
                            "asset_id": asset_id,
                        }
                    )
        return algorithm_asset_pairs

    def _delete_module(
        self,
        current_algorithm_key: str,
        module_id: str,
        algorithm_module_pairs: list[dict],
    ) -> None:
        """
        Delete a module from the database. Before deleting the module, it checks if
        the module is used by any other algorithm. If it is, it does not delete the module.

        Parameters
        ----------
        current_algorithm_key : str
            The key of the algorithm that is requesting the module deletion.
        module_id : str
            The id of the module to delete.
        algorithm_module_pairs : list[dict]
            A list of all algorithm-module pairs in the database.
        Returns
        -------
        None
        """

        module_used = False

        for pair in algorithm_module_pairs:
            if (
                pair["module_id"] == module_id
                and pair["algorithm_key"] != current_algorithm_key
            ):
                module_used = True
                break
        if not module_used:
            self.database_connection.delete_objects(
                self.module_collection, [module_id]
            )
            self.logger.info(f"Deleted module {module_id}.")

        else:
            self.logger.info(
                f"Module {module_id} is used by other algorithms. Not deleting."
            )
        return None

    def _delete_asset(
        self,
        current_algorithm_key: str,
        asset_id: str,
        algorithm_asset_pairs: list[dict],
    ) -> None:
        """
        Delete an asset from the database. Before deleting the asset, it checks if
        the asset is used by any other algorithm. If it is, it does not delete the asset.

        Parameters
        ----------
        current_algorithm_key : str
            The key of the algorithm that is requesting the asset deletion.
        asset_id : str
            The id of the asset to delete.
        algorithm_asset_pairs : list[dict]
            A list of all algorithm-asset pairs in the database.
        Returns
        -------
        None
        """

        asset_used = False

        for pair in algorithm_asset_pairs:
            if (
                pair["asset_id"] == asset_id
                and pair["algorithm_key"] != current_algorithm_key
            ):
                asset_used = True
                break
        if not asset_used:
            self.database_connection.delete_objects(
                self.assets_collection, [asset_id]
            )
            self.logger.info(f"Deleted asset {asset_id}.")
        else:
            self.logger.info(
                f"Asset {asset_id} is used by other algorithms. Not deleting."
            )
        return None
