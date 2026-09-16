"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import uuid
import hashlib
import functools
import weakref
import os
import subprocess
import importlib.util
import re
import io
import signal
from collections import deque
from functools import partial
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from compox.database_connection import BaseConnection

if TYPE_CHECKING:
    from compox.internal.JobPOpen import JobPOpen


def check_system_gpu_availability() -> tuple[bool | None, int | None]:
    """
    Check if system has GPU support.

    Returns
    -------
    bool | None
        True if CUDA is available, False otherwise.
    int | None
        The number of available GPUs.
    """
    try:
        process_fn = get_subprocess_fn()
        process = process_fn(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stdout=subprocess.PIPE,
        )
        result, error = process.communicate()
        if process.returncode:
            raise RuntimeError(
                f"Calling nvidia-smi failed with error: {error}!"
            )
        devices = []
        for line in result.strip().split("\n"):
            index, name, memory = re.split(r",\s*", line)
            devices.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_total": int(memory),  # Memory in MB
                }
            )

        cuda_available = True
        cuda_capable_devices_count = len(devices)
    except Exception as _:
        cuda_available = None
        cuda_capable_devices_count = None

    return cuda_available, cuda_capable_devices_count


def check_torch_with_cuda_available() -> bool:
    """
    Check if PyTorch has CUDA support.

    Returns
    -------
    bool
        True if PyTorch has CUDA support, False otherwise.
    """
    if importlib.util.find_spec("torch") is None:
        return False
    else:
        import torch

        return torch.cuda.is_available() and hasattr(torch, "cuda")


def check_mps_availability() -> bool:
    """
    Check if MacOS MPS (Metal Performance Shaders) is available.

    Returns
    -------
    bool
        True if MPS is available, False otherwise.
    """
    if importlib.util.find_spec("torch") is None:
        return False
    else:
        import torch

        backends = torch.backends
        mps_available = (
            hasattr(backends, "mps") and torch.backends.mps.is_available()
        )

    return mps_available


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
        If uuid version is not 1 or 4.
    """
    if version == 1:
        return str(uuid.uuid1())
    elif version == 4:
        return str(uuid.uuid4())
    else:
        raise ValueError("uuid version must be 1 or 4")


def calculate_s3_etag(bytes_obj: io.BytesIO) -> str:
    """
    Calculate the etag hash of a file, the etag should be the same as the etag
    calculate internally by the boto3/minio client

    Parameters
    ----------
    bytes_obj : io.BytesIO
        The file bytes to calculate the etag hash of.

    Returns
    -------
    str
        The etag hash.

    """
    md5s = hashlib.md5(bytes_obj.getvalue())
    return '"{}"'.format(md5s.hexdigest())


def find_algorithm_by_id(
    algorithm_id: str,
    bucket_contents: list[dict],
    separator: str = "~",
) -> tuple:
    """
    Find an algorithm by its id.

    Parameters
    ----------
    algorithm_id : str
        The id of the algorithm.
    bucket_contents : list[dict]
        The bucket contents.
    separator : str, optional
        The separator between the fields in the key. The default is "~".
    Returns
    -------
    tuple
        The algorithm key, id, name, major version, minor version.
    """
    found_model_key = None
    found_model_id = None
    found_model_name = None
    found_model_major_version = None
    found_model_minor_version = -1
    for key in bucket_contents:
        if isinstance(key, dict) and "Key" in key.keys():
            key = str(key["Key"])

        split_algorithm_key = key.split(separator)

        if len(split_algorithm_key) == 3:
            bucket_model_id = split_algorithm_key[0]
            bucket_model_name = split_algorithm_key[1]
            bucket_model_major_version = split_algorithm_key[2]
            bucket_model_minor_version = 0

        elif len(split_algorithm_key) == 4:
            bucket_model_id = split_algorithm_key[0]
            bucket_model_name = split_algorithm_key[1]
            bucket_model_major_version = split_algorithm_key[2]
            bucket_model_minor_version = split_algorithm_key[3]
        if bucket_model_id == algorithm_id:
            found_model_key = key
            found_model_id = bucket_model_id
            found_model_name = bucket_model_name
            found_model_major_version = int(bucket_model_major_version)
            found_model_minor_version = int(bucket_model_minor_version)
            return (
                found_model_key,
                found_model_id,
                found_model_name,
                found_model_major_version,
                found_model_minor_version,
            )
    return (
        None,
        None,
        None,
        None,
        None,
    )


def weak_lru(maxsize=128, typed=False):
    'LRU Cache decorator that keeps a weak reference to "self"'

    def wrapper(func):
        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper


def algorithm_cache(maxsize=None, maxsize_attr: str | None = None):
    """
    A cache decorator for algorithms. The cache is based on the algorithm_id and device.
    The cache is implemented as a dictionary with a maximum size. When the algorithm is requested
    the cache is checked and if the algorithm with the same algorithm_id and device is found
    the algorithm's Runner object is returned from the cache. If the algorithm is not found in the
    cache the algorithm is executed and the result is stored in the cache. If the cache size limit
    is reached the oldest cache entry is invalidated.

    Parameters
    ----------
    maxsize : int, optional
        The maximum size of the cache. The default is None.
    maxsize_attr : str, optional
        The attribute name to get the maximum size from the class. The default is None.

    Returns
    -------
    function
        The decorated function.
    """
    cache = {}
    access_order = deque()

    def wrapper(func):
        def inner_wrapper(self, *args):
            # Generate a unique key based on the method name and arguments

            key = tuple(repr(arg) for arg in args)
            current_maxsize = (
                getattr(type(self), maxsize_attr, maxsize)
                if maxsize_attr is not None
                else maxsize
            )

            if key in cache:
                # Update access order
                access_order.remove(key)
                access_order.append(key)
                return cache[key]
            else:
                result = func(self, *args)
                cache[key] = result
                access_order.append(key)

                # Check if cache size limit is reached
                while (
                    current_maxsize is not None
                    and current_maxsize >= 0
                    and len(cache) > current_maxsize
                ):
                    # Invalidate the oldest cache entry
                    oldest_key = access_order.popleft()
                    del cache[oldest_key]

                return result

        return inner_wrapper

    return wrapper


def data_cache(maxsize=None):
    """
        A cache decorator for data. The cache is based on the unique file key.
        The cache is implemented as a dictionary with a maximum size. When the file is requested
        the cache is checked and if the file with the same key is found
        the file is returned from the cache. If the file is not found in the
        cache the file is read and the result is stored in the cache. If the cache size limit
        is reached the oldest cache entry is invalidated.

        Parameters
        ----------
        maxsize : int, optional
            The maximum size of the cache. The default is None.
    ;
    """
    cache = {}
    access_order = deque()

    def wrapper(func):
        def inner_wrapper(self, *args):
            # Generate a unique key based on the method name and arguments

            key = tuple(repr(arg) for arg in args)

            if key in cache:
                # Update access order
                access_order.remove(key)
                access_order.append(key)
                return cache[key]
            else:
                result = func(self, *args)
                cache[key] = result
                access_order.append(key)

                # Check if cache size limit is reached
                if maxsize is not None and len(cache) > maxsize:
                    # Invalidate the oldest cache entry
                    oldest_key = access_order.popleft()
                    del cache[oldest_key]

                return result

        return inner_wrapper

    return wrapper


def check_and_create_database_collections(
    collection_names: list[str],
    database_connection: BaseConnection.BaseConnection,
) -> list[str]:
    """
    Checks if the collections exist in the database and creates them if they do not exist.

    Parameters
    ----------
    collection_names : list[str]
        The collection names.

    database_connection : BaseConnection.BaseConnection
        The database connection object.

    Returns
    -------
    list[str]
        The list of newly created collections.
    """
    collections_exist = database_connection.check_collections_exists(
        collection_names
    )
    not_existing_collections = [
        collection_names[i]
        for i in range(len(collection_names))
        if not collections_exist[i]
    ]
    if len(not_existing_collections) > 0:
        database_connection.create_collections(not_existing_collections)

    return not_existing_collections


def get_subprocess_fn() -> partial["JobPOpen"] | Any:
    """
    Get the subprocess function appropriate for the current operating system.

    Returns
    -------
    partial[JobPOpen.JobPOpen]
        A callable object used to launch subprocesses.

    Raises
    ------
    ValueError
        If the operating system is not supported.
    """
    if os.name == "posix":
        subprocess_fn = subprocess.Popen
    elif os.name == "nt":
        from compox.internal import JobPOpen

        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess_fn = partial(JobPOpen.JobPOpen, startupinfo=startupinfo)
    else:
        raise ValueError(f"Unsupported OS: {os.name}")

    return subprocess_fn


@contextmanager
def _systray_signal_shutdown(server, systray_interface, shutdown_event=None):
    """
    Bridge terminal signals to systray/server shutdown so Ctrl+C and SIGTERM
    terminate cleanly even when the systray loop is active.
    """
    previous_handlers = {}

    def _request_shutdown(signum, frame):
        server.logger.info(
            f"Received signal {signum}; requesting server and systray shutdown."
        )
        server.should_exit = True
        if shutdown_event is not None:
            shutdown_event.set()
        try:
            systray_interface.stop()
        except Exception:
            # Ignore tray stop errors; server shutdown is the priority.
            pass

    for signal_name in ("SIGINT", "SIGTERM"):
        signal_type = getattr(signal, signal_name, None)
        if signal_type is None:
            continue
        try:
            previous_handlers[signal_type] = signal.getsignal(signal_type)
            signal.signal(signal_type, _request_shutdown)
        except (ValueError, OSError, RuntimeError):
            # Some environments prevent setting process-wide signal handlers.
            continue

    try:
        yield
    finally:
        for signal_type, previous_handler in previous_handlers.items():
            try:
                signal.signal(signal_type, previous_handler)
            except (ValueError, OSError, RuntimeError):
                continue
