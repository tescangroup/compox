"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import base64
import hashlib
import time

import boto3
from loguru import logger
from botocore.config import Config
from botocore.exceptions import ClientError

from compox.database_connection.database_utils import (
    S3FileUploader,
    calculate_etag_multipart,
)
from compox.database_connection.BaseConnection import BaseConnection
from compox.database_connection.exceptions import reraised_storage_error


class S3Connection(BaseConnection):
    """
    A connection class for an S3 object storage database. This class inherits from
    the BaseConnection class and implements the methods for interacting with an S3
    object storage database.

    NOTE: All lifecycle policies are initialized at bucket creation time, this means
    that changing the expiration days in the S3Connection instance after the bucket
    creation will NOT update the lifecycle policies of already existing buckets.

    Parameters
    ----------
    endpoint_url : str
        The endpoint URL.
    aws_access_key_id : str
        The AWS access key ID.
    aws_secret_access_key : str
        The AWS secret access key.
    region_name : str | None
        The region name.
    data_store_expire_days : int
        The number of days after which the objects in the data-store bucket
        expire. Default is 1.
    execution_store_expire_days : int
        The number of days after which the objects in the execution-store bucket
        expire. Default is 30.
    training_store_expire_days : int
        The number of days after which the objects in the training-store bucket
        expire. Default is 30.
    deploy_store_expire_days : int
        The number of days after which the objects in the deploy-store bucket
        expire. Default is 30.
    stop_requests_expire_days : int
        The number of days after which the objects in the stop-requests bucket
        expire. Default is 7.
    collection_prefix : str
        The prefix for the actual bucket names. The bucket names are constructed
        as {collection_prefix}{collection_name}. Default is an empty string.
    """

    def __init__(
        self,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str | None = None,
        data_store_expire_days: int = 1,
        execution_store_expire_days: int = 30,
        training_store_expire_days: int = 30,
        deploy_store_expire_days: int = 30,
        stop_requests_expire_days: int = 7,
        collection_prefix: str = "",
    ):

        super().__init__()
        self.logger = logger.bind(
            log_type="DB",
        )
        config = Config(retries={"total_max_attempts": 20, "mode": "standard"})

        self.endpoint_url = endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            config=config,
        )
        self.s3_client.meta.events.register(
            "before-call.s3.PutBucketLifecycleConfiguration",
            self._add_content_md5_header,
        )
        self.s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            config=config,
        )
        self.region_name = region_name
        self.post_data_retries = 5
        self.post_data_retry_initial_delay = 0.1
        self.post_data_retry_max_delay = 2.0
        self.uploader = S3FileUploader(self.s3_client)
        self.data_store_expire_days = data_store_expire_days
        self.execution_store_expire_days = execution_store_expire_days
        self.training_store_expire_days = training_store_expire_days
        self.deploy_store_expire_days = deploy_store_expire_days
        self.stop_requests_expire_days = stop_requests_expire_days
        self.collection_prefix = (
            f"{collection_prefix}" if collection_prefix else ""
        )

    @staticmethod
    def _add_content_md5_header(params: dict, **kwargs) -> None:
        """
        Add Content-MD5 to lifecycle requests for S3-compatible stores.

        Recent botocore versions may use flexible checksum headers instead of
        Content-MD5. MinIO still requires Content-MD5 for bucket lifecycle
        configuration requests, so add it after botocore has serialized the XML
        body and before the request is signed and sent.
        """
        headers = params.setdefault("headers", {})
        if any(name.lower() == "content-md5" for name in headers):
            return

        body = params.get("body")
        if body is None:
            return

        if isinstance(body, str):
            body_bytes = body.encode("utf-8")
        elif isinstance(body, (bytes, bytearray)):
            body_bytes = bytes(body)
        else:
            position = body.tell()
            body_bytes = body.read()
            body.seek(position)
            if isinstance(body_bytes, str):
                body_bytes = body_bytes.encode("utf-8")

        headers["Content-MD5"] = base64.b64encode(
            hashlib.md5(body_bytes).digest()
        ).decode("ascii")

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.endpoint_url,
                self.aws_access_key_id,
                self.aws_secret_access_key,
                self.region_name,
                self.data_store_expire_days,
                self.execution_store_expire_days,
                self.training_store_expire_days,
                self.deploy_store_expire_days,
                self.stop_requests_expire_days,
                self.collection_prefix,
            ),
        )

    def list_collections(self) -> list:
        """
        Lists all collections.

        Returns
        -------
        list
            The list of collections.
        """
        return [
            bucket.name.replace(self.collection_prefix, "")
            for bucket in self.s3.buckets.all()
        ]

    def check_collections_exists(
        self, collection_names: list[str]
    ) -> list[bool]:
        """
        Checks if buckets exist.

        Parameters
        ----------
        collection_names : list[str]
            The collection names.

        Returns
        -------
        list[bool]
            The list of booleans indicating if the collection exist.
        """
        return [
            collection_name in self.list_collections()
            for collection_name in collection_names
        ]

    def delete_collections(self, collection_names: list[str]) -> None:
        """
        Deletes collections.

        Parameters
        ----------
        collection_names : list[str]
            The collection names.
        """
        for collection_name in collection_names:
            bucket = self.s3.Bucket(
                f"{self.collection_prefix}{collection_name}"
            )
            for key in bucket.objects.all():
                key.delete()
            bucket.delete()

    def create_collections(self, collection_names: list[str]) -> None:
        """
        Creates collections.

        Parameters
        ----------
        collection_names : list[str]
            The collection names.
        """
        for collection_name in collection_names:
            if self.region_name:
                self.s3.create_bucket(
                    Bucket=f"{self.collection_prefix}{collection_name}",
                    CreateBucketConfiguration={
                        "LocationConstraint": self.region_name
                    },
                )
            else:
                self.s3.create_bucket(
                    Bucket=f"{self.collection_prefix}{collection_name}"
                )

            if collection_name == "data-store":
                lifecycle_policy = {
                    "Rules": [
                        {
                            "ID": "Delete objects after expiration days",
                            "Filter": {
                                "Tag": {"Key": "training_ref", "Value": "0"}
                            },
                            "Status": "Enabled",
                            "Expiration": {"Days": self.data_store_expire_days},
                        }
                    ]
                }
                self.s3_client.put_bucket_lifecycle_configuration(
                    Bucket=f"{self.collection_prefix}{collection_name}",
                    LifecycleConfiguration=lifecycle_policy,
                )
            elif collection_name == "execution-store":
                lifecycle_policy = {
                    "Rules": [
                        {
                            "ID": "Delete objects after expiration days",
                            "Filter": {"Prefix": ""},  # applies to all objects
                            "Status": "Enabled",
                            "Expiration": {
                                "Days": self.execution_store_expire_days
                            },
                        }
                    ]
                }
                self.s3_client.put_bucket_lifecycle_configuration(
                    Bucket=f"{self.collection_prefix}{collection_name}",
                    LifecycleConfiguration=lifecycle_policy,
                )
            elif collection_name == "training-store":
                lifecycle_policy = {
                    "Rules": [
                        {
                            "ID": "Delete objects after expiration days",
                            "Filter": {"Prefix": ""},  # applies to all objects
                            "Status": "Enabled",
                            "Expiration": {
                                "Days": self.training_store_expire_days
                            },
                        }
                    ]
                }
                self.s3_client.put_bucket_lifecycle_configuration(
                    Bucket=f"{self.collection_prefix}{collection_name}",
                    LifecycleConfiguration=lifecycle_policy,
                )
            elif collection_name == "deploy-store":
                lifecycle_policy = {
                    "Rules": [
                        {
                            "ID": "Delete objects after expiration days",
                            "Filter": {"Prefix": ""},  # applies to all objects
                            "Status": "Enabled",
                            "Expiration": {
                                "Days": self.deploy_store_expire_days
                            },
                        }
                    ]
                }
                self.s3_client.put_bucket_lifecycle_configuration(
                    Bucket=f"{self.collection_prefix}{collection_name}",
                    LifecycleConfiguration=lifecycle_policy,
                )
            elif collection_name == "stop-requests":
                lifecycle_policy = {
                    "Rules": [
                        {
                            "ID": "Delete objects after expiration days",
                            "Filter": {"Prefix": ""},  # applies to all objects
                            "Status": "Enabled",
                            "Expiration": {
                                "Days": self.stop_requests_expire_days
                            },
                        }
                    ]
                }
                self.s3_client.put_bucket_lifecycle_configuration(
                    Bucket=f"{self.collection_prefix}{collection_name}",
                    LifecycleConfiguration=lifecycle_policy,
                )

    def list_objects(self, collection_name: str) -> list[dict]:
        """
        Lists all objects in a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.

        Returns
        -------
        list[dict]
            The list of object keys.
        """

        listed = self.s3_client.list_objects_v2(
            Bucket=f"{self.collection_prefix}{collection_name}"
        )

        if "Contents" not in listed:
            return []
        else:
            return listed["Contents"]

    def check_objects_exist(
        self, collection_name: str, object_names: list[str]
    ) -> list[bool]:
        """
        Checks if objects exist in a bucket.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_names : list[str]
            The object keys.

        Returns
        -------
        list[bool]
            The list of booleans indicating if the objects exist.
        """
        object_exists = []
        for object_key in object_names:
            try:
                self.s3.Object(
                    f"{self.collection_prefix}{collection_name}", object_key
                ).load()
                object_exists.append(True)
            except Exception as _:
                object_exists.append(False)

        return object_exists

    def delete_objects(
        self, collection_name: str, object_names: list[str]
    ) -> None:
        """
        Deletes objects in a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_names : list[str]
            The object keys.
        """
        for object_key in object_names:
            self.s3.Object(
                f"{self.collection_prefix}{collection_name}", object_key
            ).delete()

    def get_objects(
        self, collection_name: str, object_names: list[str]
    ) -> list[bytes]:
        """
        Gets objects from a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_names : list[str]
            The object keys.

        Returns
        -------
        list[bytes]
            The list of object bytes.
        """
        objects = []
        for object_key in object_names:
            obj = self.s3.Object(
                f"{self.collection_prefix}{collection_name}", object_key
            )
            objects.append(obj.get()["Body"].read())
        return objects

    def get_object_sizes(
        self, collection_name: str, object_names: list[str]
    ) -> list[int]:
        """
        Get object sizes in bytes without downloading object bodies.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_names : list[str]
            The object keys.

        Returns
        -------
        list[int]
            The list of object sizes in bytes.
        """
        sizes = []
        for object_key in object_names:
            response = self.s3_client.head_object(
                Bucket=f"{self.collection_prefix}{collection_name}",
                Key=object_key,
            )
            sizes.append(int(response["ContentLength"]))
        return sizes

    def put_objects(
        self,
        collection_name: str,
        object_names: list[str],
        object: list[bytes] | list[str],
    ) -> None:
        """
        Puts objects into a collection.

        Parameters
        ----------
        collection_name : str
            The collection name.
        object_names : list[str]
            The object keys.
        object : list[bytes] | list[str]
            The byte objects.
        """
        try:
            for i in range(len(object_names)):
                if (
                    not hasattr(object[i], "__len__")
                    or len(object[i]) < self.uploader.chunk_size
                ):
                    # if the object is smaller than the chunk size, upload it in one go
                    last_error = None
                    try:
                        self.s3_client.put_object(
                            Body=object[i],
                            Bucket=f"{self.collection_prefix}{collection_name}",
                            Key=object_names[i],
                        )
                    except ClientError as e:
                        last_error = e
                        # this handles the occasional
                        # botocore.exceptions.ClientError: An error occurred (AccessDenied)
                        # when calling the PutObject operation: Access Denied.
                        # and retry the upload
                        for retry_index in range(self.post_data_retries):
                            retry_delay = min(
                                self.post_data_retry_initial_delay
                                * (2 ** retry_index),
                                self.post_data_retry_max_delay,
                            )
                            time.sleep(retry_delay)
                            try:
                                self.s3_client.put_object(
                                    Body=object[i],
                                    Bucket=f"{self.collection_prefix}{collection_name}",
                                    Key=object_names[i],
                                )
                                break
                            except ClientError as e:
                                last_error = e
                                self.logger.error(
                                    f"Error uploading object {object_names[i]}: {e}"
                                )
                                continue
                        else:
                            raise last_error
                else:
                    # if the object is larger than the chunk size, upload it in parts
                    self.uploader.upload_file_multipart(
                        object[i],
                        object_names[i],
                        f"{self.collection_prefix}{collection_name}",
                    )
                if collection_name == "data-store":
                    tags = self.get_object_tags(
                        collection_name, object_names[i]
                    )
                    if tags.get("training_ref") is None:
                        tags["training_ref"] = "0"
                        self.put_object_tags(
                            collection_name, object_names[i], tags
                        )
        except Exception as exc:
            raise reraised_storage_error(
                exc,
                operation=f"put_objects to {collection_name}",
            ) from exc

    def put_objects_with_duplicity_check(
        self, collection_name: str, object_names: list[str], object: list[bytes]
    ) -> list[str]:
        """
        Puts objects into a collection with duplicity check. Returns the list of
        object keys, where the objects of which duplicates were found, substituted
        with the object keys of the duplicates. The check is based on the ETag.

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
        list[str]
            The list of object names.
        """

        for i in range(len(object_names)):

            # get all etags of the objects in the bucket
            paginator = self.s3_client.get_paginator("list_objects")
            page_iterator = paginator.paginate(
                Bucket=f"{self.collection_prefix}{collection_name}"
            )
            etags = []
            keys = []
            for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        etags.append(obj["ETag"])
                        keys.append(obj["Key"])

            # calculate etag
            etag = calculate_etag_multipart(object[i], self.uploader.chunk_size)

            if etag in etags:
                # if etag exists, get the object name
                self.logger.info(
                    "Existing file found with matching etag, returning its key..."
                )
                object_names[i] = keys[etags.index(etag)]
            else:
                # if etag does not exist, upload the object
                self.uploader.upload_file_multipart(
                    object[i],
                    object_names[i],
                    f"{self.collection_prefix}{collection_name}",
                )

        return object_names

    def get_presigned_download_url(
        self, collection_name: str, object_name: str, expiration: int = 3600
    ) -> str:
        """
        Generate a presigned URL for downloading an object.

        Parameters
        ----------
        collection_name : str
            The name of the bucket where the object is stored.
        object_name : str
            The key of the object in the bucket.
        expiration : int, optional
            Time in seconds until the URL expires.

        Returns
        -------
        str
            A presigned URL that can be used to download the object.
        """
        return self.generate_presigned_url(
            "get_object", collection_name, object_name, expiration
        )

    def get_presigned_upload_url(
        self, collection_name: str, object_name: str, expiration: int = 3600
    ) -> str:
        """
        Generate a presigned URL for uploading an object.

        Parameters
        ----------
        collection_name : str
            The name of the bucket where the object will be stored.
        object_name : str
            The key of the object in the bucket.
        expiration : int, optional
            Time in seconds until the URL expires.

        Returns
        -------
        str
            A presigned URL that can be used to upload the object.
        """
        return self.generate_presigned_url(
            "put_object", collection_name, object_name, expiration
        )

    def generate_presigned_url(
        self,
        client_method: str,
        collection_name: str,
        object_name: str,
        expiration: int = 3600,
    ) -> str:
        """
        Generate a generic presigned URL.

        Parameters
        ----------
        client_method : str
            The S3 client method to use (e.g., 'get_object' or 'put_object').
        collection_name : str
            The name of the bucket where the object will be stored.
        object_name : str
            The key of the object in the bucket.
        expiration : int, optional
            Time in seconds until the URL expires.

        Returns
        -------
        str
            A presigned URL.
        """
        return self.s3_client.generate_presigned_url(
            client_method,
            Params={
                "Bucket": f"{self.collection_prefix}{collection_name}",
                "Key": object_name,
            },
            ExpiresIn=expiration,
        )

    def get_object_tags(
        self, collection_name: str, object_name: str
    ) -> dict[str, str]:
        resp = self.s3_client.get_object_tagging(
            Bucket=f"{self.collection_prefix}{collection_name}",
            Key=object_name,
        )
        return {tag["Key"]: tag["Value"] for tag in resp.get("TagSet", [])}

    def put_object_tags(
        self, collection_name: str, object_name: str, tags: dict[str, str]
    ) -> None:
        tagset = [{"Key": k, "Value": str(v)} for k, v in tags.items()]
        self.s3_client.put_object_tagging(
            Bucket=f"{self.collection_prefix}{collection_name}",
            Key=object_name,
            Tagging={"TagSet": tagset},
        )
