"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from uuid import UUID
import os
import io
import h5py
import numpy as np
from PIL import Image
import requests
import socket


def is_valid_uuid(uuid_to_test: str, version: int = 1) -> bool:
    """
    Check if uuid_to_test is a valid UUID.
    Parameters
    ----------
    uuid_to_test : str
        The uuid to test.
    version : int, optional
        The uuid version. The default is 1.
    Returns
    -------
    bool
        Whether uuid_to_test is a valid UUID.
    """

    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test


def prepare_payload(image_stack_path: str) -> io.BytesIO:
    """
    Prepare payload for post request.
    Parameters
    ----------
    image_stack_path : str
        The path to the image stack.
    Returns
    -------
    io.BytesIO
        The payload.

    """
    image_paths = [
        os.path.join(image_stack_path, file)
        for file in os.listdir(image_stack_path)
        if file.endswith(".png")
    ]
    # sort the image paths by filename to ensure correct order in the stack
    image_paths.sort()
    images = [np.array(Image.open(image_path)) for image_path in image_paths]
    images = np.array(images)
    print(images.shape)
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        f["image_stack"] = images
    bio.seek(0)
    return bio


def prepare_random_payload(
    num_of_slices: int, width: int, height: int
) -> io.BytesIO:
    """
    Prepare payload for post request.
    Parameters
    ----------
    num_of_slices : int
        The number of slices.
    width : int
        The width of the image.
    height : int
        The height of the image.
    Returns
    -------
    io.BytesIO
        The payload.

    """

    images = np.random.randint(
        0, 255, (num_of_slices, width, height), dtype=np.uint8
    )
    print(images.shape)
    files = []
    for i in range(images.shape[0]):
        bio = io.BytesIO()
        with h5py.File(bio, "w") as f:
            f["image"] = images[i]
        bio.seek(0)
        files.append(bio)
    return files


def prepare_sample(
    input_files: list[str], target_files: list[str], tags: list[str] = []
) -> list[dict[str]]:
    """
    Prepare sample payload with simple input-target file paring schema.
    Parameters
    ----------
    input_files : list[str]
        List of input file hashes.
    target_files : list[str]
        List of target file hashes.
    tags : list[str]
        Tags assigned to the sample.
    Returns
    -------
    list[dict[str]]
        The payload.
    """
    min_list_size = min(len(input_files), len(target_files))
    files = []
    for i in range(min_list_size):
        files.append({"input": [input_files[i]], "target": [target_files[i]]})
    if not tags:
        payload = {
            "files": files,
        }
    else:
        payload = {
            "files": files,
            "tags": tags,
        }
    return [payload]


# Helper function to perform API call
def get_algorithm_id(
    endpoint_url: str,
    name: str,
    version: str,
    headers=None,
    use_name: bool = True,
    use_version: bool = True,
) -> requests.Response:
    """
    Get algorithm id.
    Parameters
    ----------
    endpoint_url : str
        The endpoint url.
    name : str
        The name of the algorithm.
    version : str
        The version of the algorithm.
    headers : dict, optional
        The headers. The default is None.
    use_name : bool, optional
        Whether to use the name. The default is True.
    use_version : bool, optional
        Whether to use the version. The default is True.
    Returns
    -------
    requests.Response
        The response.
    """

    if headers is not None:
        if use_name and use_version:
            response = requests.get(
                f"{endpoint_url}/{name}/{version}", headers=headers
            )
        elif use_name:
            response = requests.get(f"{endpoint_url}/{name}", headers=headers)
        elif use_version:
            response = requests.get(
                f"{endpoint_url}/{version}", headers=headers
            )
        elif not use_name and not use_version:
            response = requests.get(f"{endpoint_url}/", headers=headers)
    else:
        if use_name and use_version:
            response = requests.get(f"{endpoint_url}/{name}/{version}")
        elif use_name:
            response = requests.get(f"{endpoint_url}/{name}")
        elif use_version:
            response = requests.get(f"{endpoint_url}/{version}")
        elif not use_name and not use_version:
            response = requests.get(f"{endpoint_url}/")
    return response


def get_all_algorithms(endpoint_url: str, headers=None) -> requests.Response:
    """
    Get all algorithms.
    Parameters
    ----------
    endpoint_url : str
        The endpoint url.
    headers : dict, optional
        The headers. The default is None.
    Returns
    -------
    requests.Response
        The response.
    """

    if headers is not None:
        response = requests.get(f"{endpoint_url}", headers=headers)
    else:
        response = requests.get(f"{endpoint_url}")
    return response


def post_objects(
    endpoint_url: str,
    payload: list[io.BytesIO] | list[str] | list[dict],
    headers=None,
) -> list[requests.Response]:
    """
    Post objects.
    Parameters
    ----------
    endpoint_url : str
        The endpoint url.
    payload : list[io.BytesIO] | list[str] | list[dict]
        The payload.
    headers : dict, optional
        The headers. The default is None.
    Returns
    -------
    requests.Response
        The response.
    """
    responses = []
    for item in payload:
        if isinstance(item, dict):
            if headers is not None:
                response = requests.post(
                    endpoint_url, headers=headers, json=item
                )
            else:
                response = requests.post(endpoint_url, json=item)
        else:
            if headers is not None:
                response = requests.post(
                    endpoint_url, headers=headers, data=item
                )
            else:
                response = requests.post(endpoint_url, data=item)
        responses.append(response)
    return responses


# function aliases
post_files = post_objects
post_samples = post_objects


def get_object(
    endpoint_url: str, object_id: str, headers=None
) -> requests.Response:
    """
    Get object.
    Parameters
    ----------
    endpoint_url : str
        The endpoint url.
    object_id : str
        The object id.
    headers : dict, optional
        The headers. The default is None.
    Returns
    -------
    requests.Response
        The response.
    """

    if headers is not None:
        response = requests.get(f"{endpoint_url}/{object_id}", headers=headers)
    else:
        response = requests.get(f"{endpoint_url}/{object_id}")
    return response


# function aliases
get_file = get_object
get_sample = get_object


def delete_object(
    endpoint_url: str, object_id: str, headers=None
) -> requests.Response:
    """
    Delete object.
    Parameters
    ----------
    endpoint_url : str
        The endpoint url.
    object_id : str
        The object id.
    headers : dict, optional
        The headers. The default is None.
    Returns
    -------
    requests.Response
        The response.
    """
    if headers is not None:
        response = requests.delete(
            f"{endpoint_url}/{object_id}", headers=headers
        )
    else:
        response = requests.delete(f"{endpoint_url}/{object_id}")
    return response


# function aliases
delete_file = delete_object
delete_sample = delete_object
delete_trained_algorithm = delete_object


def execute_algorithm(
    endpoint_url: str,
    input_dataset_ids: list[str] = None,
    algorithm_id: str = None,
    headers=None,
) -> requests.Response:
    """
    Execute algorithm.
    Parameters
    ----------
    endpoint_url : str
        The endpoint url.
    input_dataset_ids : list[str]
        The input dataset id.
    algorithm_id : str
        The algorithm id.
    headers : dict, optional
        The headers. The default is None.
    Returns
    -------
    requests.Response
        The response.
    """
    if input_dataset_ids is None:
        payload = {
            "algorithm_id": algorithm_id,
        }
    elif algorithm_id is None:
        payload = {
            "input_dataset_ids": input_dataset_ids,
        }
    else:
        payload = {
            "input_dataset_ids": input_dataset_ids,
            "algorithm_id": algorithm_id,
        }
    if headers is not None:
        response = requests.post(endpoint_url, headers=headers, json=payload)
    else:
        response = requests.post(endpoint_url, json=payload)
    return response


def stop_algorithm_execution(
    endpoint_url: str,
    execution_id: str,
    headers=None,
) -> requests.Response:
    """
    Stop algorithm execution.
    Parameters
    ----------
    endpoint_url : str
        The endpoint url.
    execution_id : str
        The execution id.
    headers : dict, optional
        The headers. The default is None.
    Returns
    -------
    requests.Response
        The response.
    """
    if headers is not None:
        response = requests.post(
            f"{endpoint_url}/{execution_id}/stop", headers=headers
        )
    else:
        response = requests.post(f"{endpoint_url}/{execution_id}/stop")
    return response


def get_process_record(
    endpoint_url: str, process_id: str, headers=None
) -> requests.Response:
    """
    Get process record.
    Parameters
    ----------
    endpoint_url : str
        The endpoint url.
    process_id : str
        The process id.
    headers : dict, optional
        The headers. The default is None.
    Returns
    -------
    requests.Response
        The response.
    """
    if headers is not None:
        response = requests.get(f"{endpoint_url}/{process_id}", headers=headers)
    else:
        response = requests.get(f"{endpoint_url}/{process_id}")
    return response


# function aliases
get_execution_record = get_process_record
get_training_record = get_process_record


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def benchmark_algorithm(
    endpoint_url: str,
    algorithm_id: str = None,
    additional_parameters: dict = None,
    headers=None,
) -> requests.Response:
    """
    Trigger an algorithm benchmark.

    Parameters
    ----------
    endpoint_url : str
        The endpoint url.
    algorithm_id : str
        The algorithm id.
    additional_parameters : dict, optional
        Additional benchmark parameters.
    headers : dict, optional
        The headers. The default is None.

    Returns
    -------
    requests.Response
        The response.
    """
    if algorithm_id is None:
        payload = {}
    else:
        payload = {"algorithm_id": algorithm_id}
    if additional_parameters is not None:
        payload["additional_parameters"] = additional_parameters
    if headers is not None:
        response = requests.post(endpoint_url, headers=headers, json=payload)
    else:
        response = requests.post(endpoint_url, json=payload)
    return response


get_benchmark_record = get_process_record


def train_algorithm(
    endpoint_url: str,
    algorithm_id: str = None,
    training_data: list[str] = None,
    addition_parameters: dict = None,
    headers=None,
) -> requests.Response:
    """
    Train algorithm.
    Parameters
    ----------
    endpoint_url : str
        The endpoint url.
    algorithm_id : str
        The algorithm id.
    training_data : list[str]
        The input training sample ids.
    addition_parameters: dict
        Addition training parameters.
    headers : dict, optional
        The headers. The default is None.
    Returns
    -------
    requests.Response
        The response.
    """
    keys = ["algorithm_id", "training_data", "addition_parameters"]
    values = [algorithm_id, training_data, addition_parameters]
    payload = dict([(k, v) for k, v in zip(keys, values) if v is not None])
    if headers is not None:
        response = requests.post(endpoint_url, headers=headers, json=payload)
    else:
        response = requests.post(endpoint_url, json=payload)
    return response
