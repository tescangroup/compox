"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict

from compox.algorithm_utils.AlgorithmConfigSchema import (
    AdditionalParameterSchema,
)


class Algorithm(BaseModel):
    """
    Algorithm model.

    Attributes
    ----------
    algorithm_name : str
        The name of the algorithm.
    algorithm_major_version : str
        The major version of the algorithm.
    """

    algorithm_name: str
    algorithm_major_version: str


class FileUploadBody(BaseModel):
    """
    File upload body model.

    Attributes
    ----------
    file_body : List
        The file body.
    """

    file_body: List


class FileUploadResponse(BaseModel):
    """
    File upload response model.

    Attributes
    ----------
    file_id : str
        The id of the file.
    """

    file_id: str


class AlgorithmRegisteredResponse(BaseModel):
    """
    Algorithm registered response model.

    Attributes
    ----------
    algorithm_id : str
        The id of the algorithm.
    algorithm_name : str
        The name of the algorithm.
    algorithm_version : str
        The major version of the algorithm.
    algorithm_minor_versions : list[str]
        The minor versions of the algorithm.
    latest_algorithm_minor_version : str
        The latest minor version of the algorithm.
    algorithm_minor_version : str
        DEPRECATED backward-compatibility field. It carries the same value as
        ``latest_algorithm_minor_version`` and is kept only for legacy clients
        that still expect ``algorithm_minor_version`` in the response.
    algorithm_input_queue : str
        The input queue of the algorithm.
    algorithm_type : str
        The type of the algorithm.
    algorithm_tags : list[str]
        The tags of the algorithm.
    algorithm_description : str
        Description of the algorithm.
    supported_devices : list[str]
        The supported devices.
    default_device : str
        The default device.
    additional_parameters : list[AdditionalParameterSchema]
        The additional parameters.
    training_parameters : list[AdditionalParameterSchema]
        The training parameters.
    removable : bool
        Whether the algorithm can be removed via the deploy delete endpoint.
    exportable : bool
        Whether the algorithm can be exported.
    checkpoints : list[str]
        The list of checkpoint ids associated with the algorithm.
    """

    algorithm_id: str
    algorithm_name: str
    algorithm_version: str
    algorithm_minor_versions: list[str]
    latest_algorithm_minor_version: str
    algorithm_minor_version: str
    algorithm_type: str
    algorithm_tags: list[str]
    algorithm_description: str
    supported_devices: list[str] = Field(default=[])
    default_device: str
    additional_parameters: list[AdditionalParameterSchema] = Field(default=[])
    training_parameters: list[AdditionalParameterSchema] = Field(default=[])
    benchmark_outputs: list[AdditionalParameterSchema] = Field(default=[])
    removable: bool = Field(default=False)
    exportable: bool = Field(default=True)
    storage_metrics: Optional[Dict[str, Union[int, str]]] = Field(default=None)


class AlgorithmDeployResponse(BaseModel):
    """
    Algorithm deploy response model.

    Attributes
    ----------
    algorithm_id : str
        The id of the algorithm.
    algorithm_name : str
        The name of the algorithm.
    algorithm_major_version : str
        The major version of the algorithm.
    algorithm_minor_version : str
        The minor version of the algorithm.
    """

    algorithm_id: str
    algorithm_name: str
    algorithm_major_version: str
    algorithm_minor_version: str


class DeployResponse(BaseModel):
    """
    Deploy response model.

    Attributes
    ----------
    deploy_id : str
        The id of the deploy job.
    """

    deploy_id: str


class DeployRecord(BaseModel):
    """
    Deploy record model.

    Attributes
    ----------
    deploy_id : str
        The id of the deploy job.
    status : str
        The status of the deploy job.
    path : str
        The local path used for deploy.
    algorithm_id : Optional[str]
        The deployed algorithm id (if available).
    algorithm_name : Optional[str]
        The deployed algorithm name (if available).
    algorithm_major_version : Optional[str]
        The deployed algorithm major version (if available).
    time_started : Optional[str]
        The time the deploy started.
    time_completed : Optional[str]
        The time the deploy completed.
    log : Optional[str]
        Error or informational log.
    """

    deploy_id: str
    status: str
    path: str
    algorithm_id: Optional[str] = Field(default=None)
    algorithm_name: Optional[str] = Field(default=None)
    algorithm_major_version: Optional[str] = Field(default=None)
    time_started: Optional[str] = Field(default=None)
    time_completed: Optional[str] = Field(default=None)
    log: Optional[str] = Field(default=None)


class FailedAlgorithmRegisteredResponse(BaseModel):
    """
    Failed algorithm response model.

    Attributes
    ----------
    algorithm_name : str
        The name of the algorithm.
    algorithm_version : str
        The version of the algorithm.
    message : str
        The message.
    """

    algorithm_name: str
    algorithm_version: str
    message: str


class IncomingExecutionRequest(BaseModel):
    """
    Incoming execution request model.

    Attributes
    ----------
    algorithm_id : str
        The id of the algorithm.
    input_dataset_ids : list[str]
        The id of the input dataset.
    checkpoint_id : str
        The id of the checkpoint, if any.
    algorithm_minor_version : str
        The minor version of the algorithm to execute.
    execution_device_override : str
        The execution device override.
    additional_parameters : dict
        The additional parameters.
    session_token : Union[str, None]
        The string identifier of the session.
    """

    algorithm_id: str
    input_dataset_ids: list[str]
    checkpoint_id: Optional[str] = Field(default=None)
    algorithm_minor_version: Optional[str] = Field(default=None)
    execution_device_override: str = Field(default=None)
    additional_parameters: dict = Field(default={})
    session_token: Union[str, None] = Field(default=None)


class ExecutionRecord(BaseModel):
    """
    Execution record model.

    Attributes
    ----------
    execution_id : str
        The id of the execution.
    algorithm_id : str
        The id of the algorithm.
    checkpoint_id : Optional[str]
        The id of the checkpoint, if any.
    algorithm_minor_version : Optional[str]
        The minor version of the executed algorithm.
    input_dataset_ids : list[str]
        The ids of the input datasets.
    execution_device_override : Optional[str]
        The requested abstract execution device class for the run, e.g. ``cpu``,
        ``gpu`` or ``mps``.
    resolved_execution_device : Optional[str]
        The concrete runtime device Compox resolved for the execution, e.g.
        ``cpu``, ``cuda`` or ``mps``.
    additional_parameters : dict
        The additional parameters.
    session_token : Union[str, None]
        The string identifier of the session.
    output_dataset_ids : list[str]
        The ids of the output datasets.
    status : str
        The status of the execution.
    progress : float
        The progress of the execution.
    time_started : str
        The time the execution started.
    time_completed : str
        The time the execution completed.
    log : str
        The log of the execution.
    """

    execution_id: str
    algorithm_id: str
    checkpoint_id: Optional[str] = Field(default=None)
    algorithm_minor_version: Optional[str] = Field(default=None)
    input_dataset_ids: list[str]
    execution_device_override: Optional[str] = Field(default=None)
    resolved_execution_device: Optional[str] = Field(default=None)
    additional_parameters: dict
    session_token: Union[str, None]
    output_dataset_ids: list[str]
    status: str
    progress: float
    time_started: str
    time_completed: str
    log: str


class ExecutionResponse(BaseModel):
    """
    Execution response model.

    Attributes
    ----------
    execution_id : str
        The id of the execution.
    """

    execution_id: str


class ExecutionLogRecord(BaseModel):
    """
    Execution log record model.

    Attributes
    ----------
    log : str
        The log.
    """

    log: str


class IncomingBenchmarkRequest(BaseModel):
    """
    Incoming benchmark request model.

    Attributes
    ----------
    algorithm_id : str
        The id of the algorithm to benchmark.
    additional_parameters : dict
        Additional parameters forwarded to the algorithm's benchmark method.
    """

    algorithm_id: str
    additional_parameters: dict = Field(default_factory=dict)


class BenchmarkRecord(BaseModel):
    """
    Benchmark record model.

    Attributes
    ----------
    benchmark_id : str
        The id of the benchmark job.
    algorithm_id : str
        The id of the algorithm.
    additional_parameters : dict
        Additional parameters forwarded to the algorithm's benchmark method.
    status : str
        The status of the benchmark.
    time_started : str
        The time the benchmark started.
    time_completed : str
        The time the benchmark completed.
    log : str
        The log of the benchmark.
    data : dict
        The benchmark result payload.
    """

    benchmark_id: str
    algorithm_id: str
    additional_parameters: dict = Field(default_factory=dict)
    status: str = Field(default="PENDING")
    time_started: str = Field(default="")
    time_completed: str = Field(default="")
    log: str = Field(default="")
    data: dict = Field(default_factory=dict)


class BenchmarkResponse(BaseModel):
    """
    Benchmark response model.

    Attributes
    ----------
    benchmark_id : str
        The id of the benchmark job.
    """

    benchmark_id: str


class IncomingTrainingRequest(BaseModel):
    """
    Incoming training request model.

    Attributes
    ----------
    algorithm_id : str
        The id of the algorithm to train.
    training_data : list[str]
        List of sample ids used as training data.
    checkpoint_id : str, optional
        The id of the input checkpoint, if any.
    algorithm_minor_version : str, optional
        The minor version of the algorithm to train.
    tags : list[str]
        The list of tags associated with the training run.
    additional_parameters : dict
        Additional training parameters (e.g., iterations, learning rate, ...).
    """

    algorithm_id: str
    training_data: list[str]
    checkpoint_id: Optional[Union[str, None]] = Field(default=None)
    algorithm_minor_version: Optional[str] = Field(default=None)
    tags: list[str] = Field(default=[])
    additional_parameters: Optional[Union[dict, None]] = Field(default=None)


class TrainingResponse(BaseModel):
    """
    Training response model.

    Attributes
    ----------
    training_id : str
        The id of the training job.
    """

    training_id: str


class TrainingRecord(BaseModel):
    """
    Training record model.

    Attributes
    ----------
    training_id: str
        The id of the training.
    status : str
        The status of the training (e.g., running, completed, failed).
    progress : float
        The progress of the training in range [0.0–1.0].
    time_started : str
        The time the training started.
    time_completed : str, optional
        The time the training completed, if available.
    log : str, optional
        The log output from the training.
    training_data : list[str]
        The list of sample ids used for training.
    state : dict
        Training state information, including metrics and losses.
    output_checkpoint_ids : list[str]
        The list of produced checkpoint ids.
    tags : list[str]
        The list of tags associated with the training run.
    checkpoint_id : str, optional
        The id of the input checkpoint, if any.
    algorithm_minor_version : str, optional
        The minor version of the algorithm to train.
    """

    training_id: str
    algorithm_id: str
    status: str
    progress: float
    time_started: str
    time_completed: Optional[str] = Field(default=None)
    log: Optional[str] = Field(default=None)
    training_data: list[str]
    additional_parameters: Optional[Union[dict, None]] = Field(default=None)
    state: dict
    tags: List[str] = Field(default=[])
    checkpoint_id: Optional[str] = Field(default=None)
    algorithm_minor_version: Optional[str] = Field(default=None)
    output_checkpoint_ids: List[str] = Field(default_factory=list)


class AlgorithmCheckpointResponse(BaseModel):
    """
    Algorithm checkpoint response model.

    Attributes
    ----------
    checkpoint_id : str
        The id of the checkpoint.
    """

    checkpoint_id: str


class AlgorithmCheckpointRecord(BaseModel):
    """
    Algorithm checkpoint record model.

    Attributes
    ----------
    checkpoint_id : str
        The id of the checkpoint.
    training_id : str
        The id of the training run that produced this checkpoint.
    parent_id : str
        The id of the parent checkpoint, if any.
    created_at : str
        The time the checkpoint was created.
    properties : dict
        A dictionary of arbitrary properties associated with the checkpoint.
    tags : list[str]
        A list of tags associated with the checkpoint.
    parent_checkpoint_id : Optional[str]
        The id of the parent checkpoint, if any.
    """

    checkpoint_id: str
    training_id: str
    parent_algorithm_id: str
    created_at: str
    properties: dict
    tags: list[str] = Field(default=[])
    parent_checkpoint_id: Optional[str] = Field(default=None)


class IncomingSampleRequest(BaseModel):
    """
    Incoming sample request model.

    Attributes
    ----------
    files : list[dict[str, list[str]]]
        The list of dicts with file paring structure.
    tags : list[str]
        The tags associated with the sample.
    """

    files: list[dict[str, list[str]]]
    tags: list[str] = Field(default=[])


class SampleResponse(BaseModel):
    """
    Sample response model.

    Attributes
    ----------
    sample_id : str
        The id of the sample.
    """

    sample_id: str


class SampleRecord(BaseModel):
    """
    Sample record model.

    Attributes
    ----------
    sample_id : str
        The id of the sample.
    files : list[dict[str, list[str]]]
        The list of dicts with file paring structure.
    tags : list[str]
        The tags associated with the sample.
    time_created : str
        The time the sample was created.
    """

    sample_id: str
    files: list[dict[str, list[str]]]
    tags: list[str] = Field(default=[])
    time_created: str


class MinioServer(BaseModel):
    """
    Minio server model.

    Attributes
    ----------
    executable_path : str
        The path to the minio executable.
    storage_path : str
        The path to the minio storage.
    console_address : str
        The address of the minio console.
    address : str
        The address of the minio server.
    """

    executable_path: str
    storage_path: str
    console_address: str
    address: str


class MinioServerInfo(BaseModel):
    """
    Minio server info model.

    Attributes
    ----------
    storage_path : str
        The path to the minio storage.
    console_address : str
        The address of the minio console.
    address : str
        The address of the minio server.
    """

    storage_path: str
    console_address: str
    address: str


class S3Bucket(BaseModel):
    """
    S3 bucket model.

    Attributes
    ----------
    bucket_name : str
        The name of the bucket.
    """

    bucket_name: str


class S3ModelFile(BaseModel):
    """
    S3 model file model.

    Attributes
    ----------
    runner_path : str
        The path to the runner file.
    algorithm_path : str
        The path to the algorithm file.
    algorithm_name : str
        The name of the algorithm.
    algorithm_major_version : str
        The major version of the algorithm.
    algorithm_minor_version : str
    """

    runner_path: str
    algorithm_path: str
    algorithm_name: str
    algorithm_major_version: str
    algorithm_minor_version: str


class S3ModelFileRecord(BaseModel):
    """
    S3 model file record model.

    Attributes
    ----------
    algorithm_key : str
        The key of the algorithm.
    """

    algorithm_key: str


class ResponseMessage(BaseModel):
    """
    Response message model.

    Attributes
    ----------
    detail : str | None
        The message.
    """

    detail: str | None = None


class ErrorResponse(ResponseMessage):
    """
    Structured error response model.

    The inherited ``detail`` field is kept for backward compatibility with
    existing clients that only read a human-readable message.
    """

    code: str
    retryable: bool = Field(default=False)
    details: Optional[Dict] = Field(default=None)


class RootMessage(BaseModel):
    """
    Root message model.

    Attributes
    ----------
    name : str
        The name of the server.
    tags: list[str]
        The server tags.
    group : str
        The group.
    organization : str
        The organization.
    domain : str
        The domain.
    version : str
        The version.
    cuda_available : bool | None
        If cuda is available.
    cuda_capable_devices_count : int | None
        The number of cuda capable devices.
    """

    name: str
    tags: list[str]
    group: str
    organization: str
    domain: str
    version: str
    cuda_available: bool | None = None
    cuda_capable_devices_count: int | None = None


class UrlResponse(BaseModel):
    """
    Url response model.

    Attributes
    ----------
    url : str
        The url.
    """

    url: str
