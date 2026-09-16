# How to create an algorithm module
The algorithm module is a Python package that contains the algorithm code and assets. The algorithm module should be structured in a specific way in order to work properly with Compox.

See also:
- ../docs/algorithm_training_guide.md
- ../docs/training_client_workflow.md

The algorithm should be structured as follows:

```plaintext
algorithm_name/
    |-- __init__.py
    |-- Runner.py
    |-- pyproject.toml
    |-- files/
    |   |-- file1
    |   `-- file2
    `-- some_internal_submodule/
        |-- __init__.py
        |-- module1.py
        `-- module2.py
```

## The Runner.py file
The Runner.py file is a mandatory component of the algorithm module. It serves as the entry point for Compox to run the algorithm. It must define a class named `Runner`. The `Runner` class can inherit from `BaseRunner` (for generic behavior) or from a Runner class specific to the algorithm type (see below). Runner classes can be imported from the `compox.algorithm_utils` package.

**Why this exists:** Compox always loads and instantiates `Runner` as the algorithm entry point, so keeping it in a predictable location allows deployment, caching, and execution to work consistently.

### Algorithm types
The algorithm type is defined in the algorithm's `pyproject.toml` file. Your Runner inheritance should match the declared type, but Compox does not infer the type from the class. If you inherit from `BaseRunner`, set `algorithm_type = "Generic"` (or leave it as `Undefined` for development, but not for production). For example, an Image2Image algorithm receives an image as input and returns an image as output. In that case, the `pyproject.toml` file should contain:

```toml
[tool.compox]
algorithm_type = "Image2Image"
```

and the Runner should inherit from the matching Runner class:

```python
from compox.algorithm_utils.Image2ImageRunner import Image2ImageRunner

class Runner(Image2ImageRunner):
    """
    The runner class for the denoiser algorithm.
    """
```

The following algorithm types are currently supported:
- Image2Image
- Image2Embedding
- Image2Segmentation
- Image2Alignment
- Segmentation2Segmentation
- Generic

`Undefined` exists as a fallback/default but should not be used for real algorithms.

**Why this exists:** typed runners provide schema and convenience helpers so you can focus on model logic instead of wiring input/output formats. `BaseRunner` is for algorithms with custom schemas or non-standard inputs/outputs.

### Algorithm tags
The algorithm tags are a useful tool to categorize algorithms for frontend applications. Tags allow clients to assume that algorithms with the same tag follow the same input/output schemas.

**Why this exists:** clients can filter and group algorithms by capability (e.g., “denoising”), and safely assume consistent I/O across similarly tagged algorithms.

### The `preprocess`, `inference` and `postprocess` methods
The `run` method calls `preprocess`, then `inference`, then `postprocess`. Each of these methods accepts two arguments (after `self`): the input data for that stage and a dictionary of user arguments (`args`).

- `preprocess(self, input_data: dict, args: dict | None = None)` typically loads data using `fetch_data`, prepares it, and returns the result for `inference`.
- `inference(self, data: Any, args: dict | None = None)` runs the model or algorithm.
- `postprocess(self, data: Any, args: dict | None = None)` should upload output datasets using `post_data` and return a list of dataset IDs.

The `input_data` dictionary contains identifiers provided by the user (commonly `input_dataset_ids`).

**Why this exists:** separating the pipeline makes data flow and logging explicit, enables progress reporting, and allows easier debugging.

### Benchmarking (optional)
If you want to measure algorithm performance, you can implement an optional `benchmark(self, args: dict | None = None) -> dict` method on your `Runner`.

The method should run representative inference on synthetic or sample data and return a dictionary of measured metrics (for example inference time, memory usage, or VRAM usage).

If you expose benchmark metrics, declare their schema in `pyproject.toml` using `benchmark_outputs` so clients can validate and display them consistently.

**Why this exists:** benchmarking helps compare algorithm variants and hardware setups in a reproducible way.

### The `fetch_data` method for BaseRunner
`fetch_data` retrieves datasets by IDs and validates them using a Pydantic schema. It expects a list of file ID strings.

Example of fetching data:

```python
embeddings = self.fetch_data(input_data["input_dataset_ids"], EmbeddingSchema)
```

The Pydantic schemas are defined in `compox/src/compox/algorithm_utils/io_schemas.py`, but you are not required to use them. You can define your own schemas by inheriting from `DataSchema` (useful for type checking and validation).

**Why this exists:** schemas provide consistent validation and type hints for downstream code, while still allowing custom formats when needed.

### The `fetch_data` method for specific algorithm types
Runner subclasses for specific algorithm types use predefined schemas, so `fetch_data` does not take a schema argument. It still expects a list of file ID strings.

Example for Image2Image:

```python
input_data = self.fetch_data(input_data["input_dataset_ids"])
```

This fetches datasets validated against the `ImageSchema`.

### The `post_data` method for BaseRunner
`post_data` uploads output datasets and validates them with a Pydantic schema. It expects a list of dictionaries, one per output dataset.

Example of posting data:

```python
output_dataset_ids = self.post_data(output, MaskSchema)
```

### The `post_data` method for specific algorithm types
For specific algorithm types, `post_data` uses predefined schemas, so no schema argument is needed.

Example for Image2Image:

```python
output_dataset_ids = self.post_data(output)
```

### The `load_assets` method
You can override `load_assets` to load model weights or other files once and cache them on the Runner instance. Use `self.fetch_asset(...)` to load files stored in the algorithm package. Paths are **relative to the Runner module root** (e.g., `files/weights.pt`). `fetch_asset` returns an `io.BytesIO` object that you can pass to libraries like `torch.load`.

Example:

```python
state_dict_bytes = self.fetch_asset("files/vit_b.pt")
state_dict = torch.load(state_dict_bytes)
```

**Why this exists:** model weights and large resources are expensive to load, so Compox caches them on the Runner instance for reuse across requests. These attributes are locked to avoid unsafe mutation across threads.

### The `log_message` method
Log messages to Compox:

```python
self.log_message("This is an info message.", logging_level="INFO")
```

### The `set_progress` method
Report execution progress (float between 0 and 1):

```python
self.set_progress(0.5)
```

### Sessions (optional)
Executions can be associated with a `session_token` to reuse an in‑memory cache across runs. From a Runner perspective, this cache is accessed via:
- `save_item_to_session(obj, key)`
- `load_item_from_session(key)`
- `remove_item_from_session(key)`

**Why this exists:** some algorithms benefit from reusing expensive intermediates (e.g., feature caches, preprocessed inputs) across multiple executions without reloading from storage.

Notes:
- Sessions are **FastAPI background task only**. Celery mode does not support sessions.
- Sessions are in‑memory (single process) and expire after a fixed timeout, so treat them as an optimization rather than persistent storage.
- The client supplies `session_token` on execution requests; the server can also generate one when missing.

## The `pyproject.toml` file
The `pyproject.toml` file contains algorithm metadata. It must be in the algorithm root.

### Mandatory fields

```toml
[project]
name = "algorithm_name"
version = "major.minor.patch"
```

**Why this exists:** Compox uses `name` + major version to identify an algorithm line and uses minor versions to track distinct builds.

### Versioning behavior (AlgorithmDeployer)
Compox derives versioning from the `[project]` version string in `pyproject.toml`:
- **Major version** = the first segment (before the first dot)
- **Minor version** = the second segment (between first and second dot)
- **Patch version** is currently ignored by the deployer

When an algorithm is deployed, Compox searches the algorithm store for an existing record
with the same **algorithm name** and **major version**. The behavior is:
- **If found:** Compox compares the newly built module ID and assets dictionary with the
  latest stored minor version. If either differs, it inserts a new **minor version** entry
  and increments `latest_algorithm_minor_version`. If both are identical, it **does not**
  insert a new minor version.
- **If not found:** Compox creates a new algorithm record with `latest_algorithm_minor_version`
  initialized from the `project.version` minor segment, and stores the module/assets under that.

Notes:
- The stored minor versions are not the original `pyproject.toml` patch version; only the
  **major/minor** segments drive versioning.
- Re-deploying the same algorithm with identical module and assets is a no‑op for minor
  versions (no new entry is added).
- If you change only non‑code assets, a new minor version is created because the assets
  dictionary changes.

**Why this exists:** this makes deployments deterministic and deduplicated; you can update assets or code without forcing a new algorithm identity while still keeping a history of builds.

### Algorithm type, tags, description

```toml
[tool.compox]
algorithm_type = "AlgorithmType"
tags = ["tag1", "tag2"]
description = "This is a super cool algorithm."
removable = false
exportable = true
```

### Supported devices
Supported devices are a list of strings: `"cpu"`, `"gpu"`, or `"mps"`. The `default_device` must be included in `supported_devices`, otherwise validation raises an error.

```toml
supported_devices = ["cpu", "gpu"]
default_device = "cpu"
```

### Additional parameters
Additional parameters are a list of objects with `name`, `description`, and a `config` section.
You can also provide an optional `displayed_name` for a more human-friendly UI label. If omitted,
Compox derives it automatically from `name`.

```toml
additional_parameters = [
  { name = "some_string_parameter", displayed_name = "Some string parameter", description = "This parameter strings.", config = { type = "string", default = "hello", adjustable = true } },
  { name = "threshold", description = "Threshold used during inference.", config = { type = "float_range", default = 0.5, min = 0.0, max = 1.0, step = 0.05, decimal_precision = 2, adjustable = true } },
]
```

Parameter types:

| Parameter type | Configuration fields |
| --- | --- |
| string | type, default, adjustable |
| int | type, default, adjustable |
| float | type, default, adjustable, decimal_precision(optional) |
| bool | type, default, adjustable |
| int_range | type, default, min, max, step, adjustable |
| float_range | type, default, min, max, step, adjustable, decimal_precision(optional) |
| string_enum | type, default, options, adjustable |
| int_enum | type, default, options, adjustable |
| float_enum | type, default, options, adjustable, decimal_precision(optional) |
| string_list | type, default, options, adjustable |
| int_list | type, default, options, adjustable |
| float_list | type, default, options, adjustable, decimal_precision(optional) |
| bool_list | type, default, options, adjustable |

Notes:
- `displayed_name` is optional. If not provided, Compox generates one from `name` by replacing `_` and `-` with spaces and capitalizing the result.
- `decimal_precision` is optional and only valid for float-based parameter types.
- `decimal_precision` must be greater than or equal to `0`.

### Training parameters
Training parameters use the same schema as additional parameters:

```toml
training_parameters = [
  { name = "epochs", displayed_name = "Epochs", description = "Training epochs.", config = { type = "int", default = 10, adjustable = true } },
]
```

### Benchmark outputs (optional)
If your `Runner` implements a `benchmark(...)` method and returns benchmark metrics,
declare those metrics in `benchmark_outputs` so clients can validate and display them.

```toml
benchmark_outputs = [
  { name = "inference_time", description = "Time taken for inference in seconds.", config = { type = "float", decimal_precision = 4 } },
]
```

### Other fields

```toml
check_importable = false
obfuscate = true
hash_module = true (deprecated; ignored, deduplication is always on)
hash_assets = true  (deprecated; ignored, deduplication is always on)
removable = false
exportable = true
```

**Why these exist:**
- `check_importable` helps catch packaging mistakes early.
- `obfuscate` reduces casual code exposure in stored modules.
- (deprecated) `hash_module` and `hash_assets` are ignored. Deduplication by content hash is always enabled.
- `removable` controls whether the deploy delete endpoint is allowed to remove this algorithm (defaults to false).
- `exportable` controls whether the export endpoint can package this algorithm (defaults to true). If false, export returns HTTP 403.

## The `files` directory
Optional. Store data assets your algorithm needs at runtime. Load them via `self.fetch_asset(...)`.

**Why this exists:** code is zipped and cached separately from assets, so non‑Python files are stored and retrieved from the asset store by path.

## The `some_internal_submodule` directory
Optional. Include internal modules used by your Runner.

**Why this exists:** any Python modules inside the algorithm directory are packaged into the module zip, so you can keep helper code alongside your Runner.
# Example of a dummy algorithm

```plaintext
algorithm_name/
    |-- __init__.py
    |-- Runner.py
    |-- pyproject.toml
    |-- files/
    |   `-- some_heavy_model.pt
    `-- my_big_model/
        |-- __init__.py
        `-- utils.py
```

Runner example:

```python
from my_big_model.utils import MyBigModel
from compox.algorithm_utils.BaseRunner import BaseRunner
from compox.algorithm_utils.io_schemas import ImageSchema, SegmentationSchema
import numpy as np
import torch

class Runner(BaseRunner):
    """
    The runner class for the foo algorithm.
    """

    def load_assets(self):
        """
        The assets to load for the foo algorithm.
        """
        some_model = MyBigModel()
        self.log_message("Loading the Foo assets.")
        state_dict_bytes = self.fetch_asset("files/some_heavy_model.pt")
        state_dict = torch.load(state_dict_bytes)
        some_model.load_state_dict(state_dict)
        self.my_big_model = some_model

    def preprocess(self, input_data: dict, args: dict | None = None) -> np.ndarray:
        self.log_message("Preprocessing the Foo input data.")
        my_data = self.fetch_data(input_data["input_dataset_ids"], ImageSchema)
        input_array = np.array(my_data[0]["image"])
        return input_array

    def inference(self, data: np.ndarray, args: dict | None = None) -> torch.Tensor:
        self.log_message("Running the Foo inference.")
        some_user_defined_args = args.get("some_user_defined_args", None)
        if some_user_defined_args is not None:
            self.log_message(f"User defined args: {some_user_defined_args}")
        output = self.my_big_model(data, some_user_defined_args)
        self.set_progress(0.5)
        self.log_message("The Foo inference is done.")
        return output

    def postprocess(self, inference_output: torch.Tensor, args: dict | None = None) -> list[str]:
        self.log_message("Postprocessing the Foo output.")
        output = inference_output.detach().numpy()
        output_dicts = [{"mask": output}]
        output_dataset_ids = self.post_data(output_dicts, SegmentationSchema)
        return output_dataset_ids
```

pyproject.toml example:

```toml
[project]
name = "foo"
version = "0.1.0"

[tool.compox]
algorithm_type = "Generic"
tags = ["foo", "bar"]
description = "This algorithm does foo and bar."
additional_parameters = [
  { name = "some_user_defined_args", description = "This is a user defined argument.", config = { type = "string", default = "hello", adjustable = true } },
]
check_importable = false
obfuscate = true
hash_module = true
hash_assets = true
```
