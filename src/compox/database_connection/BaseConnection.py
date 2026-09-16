"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""


class BaseConnection:
    """
    A generic database connection class. This class is meant to be inherited by
    specific database connection classes. It defines the methods for interacting
    with the object storage database. It assumes that the database is structured
    as a set of collections, where each collection contains a set of
    objects. The objects can be any type of data, such as files, images, or
    other objects. The objects are acessed by their names, and the collections
    are accessed by their names. For example, in an S3 database, the collections
    would be the buckets, and the objects would be the files in the buckets.
    """

    def __init__(self):
        pass

    def list_collections(self) -> list:
        """
        Lists all object collections.

        Returns
        -------
        list
            The list of object collections.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def check_collections_exists(self, collection_names: list[str]) -> list[bool]:
        """
        Checks if collections exist.

        Parameters
        ----------
        collection_names : list[str]
            The collection names.

        Returns
        -------
        list[bool]
            The list of booleans indicating if the collections exist.
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def delete_collections(self, collection_names: list[str]) -> None:
        """
        Deletes collections.

        Parameters
        ----------
        collection_names : list[str]
            The collection names.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def create_collections(self, collection_names: list[str]) -> None:
        """
        Creates collections.

        Parameters
        ----------
        collection_names : list[str]
            The collection names.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def list_objects(self, collection_name: str) -> list[dict] | list[str]:
        """
        Lists all objects in a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.

        Returns
        -------
        list[dict] | list[str]
            The list of objects in the collection.
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def check_objects_exist(
        self, collection_name: str, object_names: list[str]
    ) -> list[bool]:
        """
        Checks if objects exist in a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_names : list[str]
            The object names.

        Returns
        -------
        list[bool]
            The list of booleans indicating if the objects exist.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def get_objects(self, collection_name: str, object_names: list[str]) -> list[bytes]:
        """
        Gets objects from a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_names : list[str]
            The object names.

        Returns
        -------
        list[bytes]
            The list of bytes objects.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def get_object_sizes(
        self, collection_name: str, object_names: list[str]
    ) -> list[int]:
        """
        Get object sizes in bytes for objects stored in a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_names : list[str]
            The object names.

        Returns
        -------
        list[int]
            The list of object sizes in bytes.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def put_objects(
        self, collection_name: str, object_names: list[str], object: list[bytes] | list[str]
    ) -> None:
        """
        Puts objects into a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_names : list[str]
            The object names.
        object : list[bytes] | list[str]
            The byte objects.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def put_objects_with_duplicity_check(
        self, collection_name: str, object_names: list[str], object: list[bytes]
    ) -> list[bool] | list[str]:
        """
        Puts objects into a collection with duplicity check. Returns the list of
        object names, where the objects of which duplicates were found, substituted
        with the object names of the duplicates.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_names : list[str]
            The object names.
        object : list[bytes]
            The byte objects.

        Returns
        -------
        list[bool] | list[str]
            The list of object names.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def delete_objects(self, collection_name: str, object_names: list[str]) -> None:
        """
        Deletes objects from a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_names : list[str]
            The object names.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def get_object_tags(self, collection_name: str, object_name: str) -> dict[str, str]:
        """
        Get object tags for a given object in a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_name : str
            The object name.

        Returns
        -------
        dict[str, str]
            The object tags.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def put_object_tags(self, collection_name: str, object_name: str, tags: dict[str, str]) -> None:
        """
        Put object tags for a given object in a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_name : str
            The object name.
        tags : dict[str, str]
            The object tags.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError
