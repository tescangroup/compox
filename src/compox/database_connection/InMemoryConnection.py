from compox.database_connection.BaseConnection import BaseConnection
from compox.database_connection.exceptions import reraised_storage_error


class InMemoryConnection(BaseConnection):
    """
    A connection class that stores data in memory for testing and debugging purposes.
    This simulates a simple object store using nested dictionaries.

    Structure:
        self.store = {
            "collection1": {
                "object1": b"...",
                "object2": b"...",
                ...
            },
            ...
        }
    """

    def __init__(self):
        super().__init__()
        self.store: dict[str, dict[str, bytes]] = {}
        self.tags: dict[str, dict[str, dict[str, str]]] = {}

    def list_collections(self) -> list[str]:
        """
        List all collection names currently in the in-memory store.

        Returns
        -------
        list[str]
            A list of collection names.
        """
        return list(self.store.keys())

    def check_collections_exists(
        self, collection_names: list[str]
    ) -> list[bool]:
        """
        Check whether each of the specified collections exists.

        Parameters
        ----------
        collection_names : list[str]
            A list of collection names to check.

        Returns
        -------
        list[bool]
            A list of booleans indicating the existence of each collection.
        """
        return [name in self.store for name in collection_names]

    def create_collections(self, collection_names: list[str]) -> None:
        """
        Create collections in the in-memory store.

        Parameters
        ----------
        collection_names : list[str]
            A list of collection names to create.
        """
        for name in collection_names:
            self.store.setdefault(name, {})

    def delete_collections(self, collection_names: list[str]) -> None:
        """
        Delete specified collections from the in-memory store.

        Parameters
        ----------
        collection_names : list[str]
            A list of collection names to delete.
        """
        for name in collection_names:
            self.store.pop(name, None)

    def list_objects(self, collection_name: str) -> list[dict] | list[str]:
        """
        List all object names within a specified collection.

        Parameters
        ----------
        collection_name : str
            The name of the collection.

        Returns
        -------
        list[dict] | list[str]
            A list of object names within the collection.
        """
        return list(self.store.get(collection_name, {}).keys())

    def check_objects_exist(
        self, collection_name: str, object_names: list[str]
    ) -> list[bool]:
        """
        Check whether each of the specified objects exists in a given collection.

        Parameters
        ----------
        collection_name : str
            The name of the collection.
        object_names : list[str]
            A list of object names to check.

        Returns
        -------
        list[bool]
            A list of booleans indicating the existence of each object.
        """
        collection = self.store.get(collection_name, {})
        return [name in collection for name in object_names]

    def delete_objects(
        self, collection_name: str, object_names: list[str]
    ) -> None:
        """
        Delete specified objects from a given collection.

        Parameters
        ----------
        collection_name : str
            The name of the collection.
        object_names : list[str]
            A list of object names to delete.
        """
        collection = self.store.get(collection_name, {})
        for name in object_names:
            collection.pop(name, None)
            self.tags.get(collection_name, {}).pop(name, None)

    def get_objects(
        self, collection_name: str, object_names: list[str]
    ) -> list[bytes]:
        """
        Retrieve specified objects from a collection.

        Parameters
        ----------
        collection_name : str
            The name of the collection.
        object_names : list[str]
            A list of object names to retrieve.

        Returns
        -------
        list[bytes]
            A list of objects in bytes.
        """
        collection = self.store.get(collection_name, {})
        return [collection[name] for name in object_names]

    def get_object_sizes(
        self, collection_name: str, object_names: list[str]
    ) -> list[int]:
        """
        Return byte sizes for the specified objects.

        Parameters
        ----------
        collection_name : str
            The name of the collection.
        object_names : list[str]
            A list of object names.

        Returns
        -------
        list[int]
            A list of object sizes in bytes.
        """
        collection = self.store.get(collection_name, {})
        sizes = []
        for name in object_names:
            obj = collection[name]
            if isinstance(obj, str):
                obj = obj.encode("utf-8")
            sizes.append(len(obj))
        return sizes

    def put_objects(
        self,
        collection_name: str,
        object_names: list[str],
        object: list[bytes] | list[str],
    ) -> None:
        """
        Store objects in the specified collection.

        Parameters
        ----------
        collection_name : str
            The name of the collection.
        object_names : list[str]
            A list of object names (keys).
        object : list[bytes] | list[str]
            A list of objects. Only `bytes` values are stored.
        """
        try:
            self.store.setdefault(collection_name, {})
            self.tags.setdefault(collection_name, {})
            for name, obj in zip(object_names, object):
                self.store[collection_name][name] = obj
                if collection_name == "data-store":
                    tags = self.tags[collection_name].get(name, {})
                    tags.setdefault("training_ref", "0")
                    self.tags[collection_name][name] = tags
        except Exception as exc:
            raise reraised_storage_error(
                exc,
                operation=f"put_objects to {collection_name}",
            ) from exc

    def get_object_tags(
        self, collection_name: str, object_name: str
    ) -> dict[str, str]:
        return self.tags.get(collection_name, {}).get(object_name, {}).copy()

    def put_object_tags(
        self, collection_name: str, object_name: str, tags: dict[str, str]
    ) -> None:
        self.tags.setdefault(collection_name, {})
        self.tags[collection_name][object_name] = {
            k: str(v) for k, v in tags.items()
        }

    def put_objects_with_duplicity_check(
        self,
        collection_name: str,
        object_names: list[str],
        object: list[bytes],
    ) -> list[bool]:
        """
        Store objects in the specified collection only if they don't already exist.

        Parameters
        ----------
        collection_name : str
            The name of the collection.
        object_names : list[str]
            A list of object names (keys).
        object : list[bytes]
            A list of objects to store.

        Returns
        -------
        list[bool]
            A list indicating whether each object already existed before insertion.
        """
        try:
            self.store.setdefault(collection_name, {})
            collection = self.store[collection_name]

            already_exists = [name in collection for name in object_names]
            for name, obj, exists in zip(object_names, object, already_exists):
                if not exists:
                    collection[name] = obj
            return already_exists
        except Exception as exc:
            raise reraised_storage_error(
                exc,
                operation=f"put_objects_with_duplicity_check to {collection_name}",
            ) from exc
