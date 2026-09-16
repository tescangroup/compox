# Generic Algorithm Template

This template is the minimal starting point for a **Generic** Compox
algorithm. Use it when the typed runners such as `Image2ImageRunner` or
`Image2SegmentationRunner` do not match your input/output contract.

If your algorithm operates on one of the standard image workflows, prefer one
of the specific templates in `compox/algorithms/`.

## Structure

```plaintext
algorithm_template/
    __init__.py
    Runner.py
    pyproject.toml
    README.md
    dependencies/
        __init__.py
        utils.py
```

## `pyproject.toml`

This template uses current Compox metadata fields:

```toml
[project]
name = "algorithm_template"
version = "1.0.0"

[tool.compox]
algorithm_type = "Generic"
tags = ["template", "generic"]
description = "Minimal generic Compox algorithm template."
supported_devices = ["cpu"]
default_device = "cpu"
additional_parameters = [
    { name = "scale", displayed_name = "Scale", description = "Multiplier applied to the input data.", config = { type = "float", default = 1.0, decimal_precision = 2, adjustable = true } },
    { name = "bias", displayed_name = "Bias", description = "Bias added after scaling.", config = { type = "float", default = 0.0, decimal_precision = 2, adjustable = true } },
]
benchmark_outputs = [
    {name = "inference_time", description = "Time taken for inference in seconds.", config = {type = "float", decimal_precision = 4}},
]
check_importable = false
obfuscate = true
```

Notes:

- `displayed_name` is optional. If omitted, Compox derives a UI label from `name`.
- `decimal_precision` is optional and only valid for float-based parameter types.

## `Runner.py`

The template runner inherits from `BaseRunner` and demonstrates the current
method contract:

- `preprocess(self, input_data: dict, args: dict | None = None)`
- `inference(self, data, args: dict | None = None)`
- `postprocess(self, data, args: dict | None = None)`

It uses:

- `GenericSchema` for input and output validation
- `fetch_data(..., GenericSchema)` to load datasets
- `post_data(..., GenericSchema)` to upload outputs
- `log_message()` and `set_progress()` during execution

The example logic is intentionally simple: it scales and shifts the input
arrays using parameters from `pyproject.toml`.

## Local helper code

The `dependencies/` package is where you can place local helper functions or
small internal modules. In this template, `dependencies/utils.py` contains the
simple transform used during inference.

## Benchmark algorithm

You can verify how computationally demanding algorithm inference is on dummy data of a specific size. You can implement the `benchmark` method and monitor inference time, VRAM usage, or other metrics.

```python
def benchmark(self, args: dict | None = None) -> dict:
    """
    Benchmark the algorithm performance.
    """
    dummy_data = np.random.rand(100, 256, 256)  # Example dummy data
    self.log_message(f"Benchmarking generic algorithm with dummy data shape: {dummy_data.shape}")
    
    start_time = time.time()
    self.inference([{"data": dummy_data}], args)
    inference_time = time.time() - start_time
    
    return {"inference_time": inference_time} 
```

## Running debug tool

You have two main options for local debugging:

- run `Runner.py` directly through the built-in `debug()` entrypoint
- run the algorithm through the CLI with `compox debug run`

### Option 1: Run `Runner.py` directly

`Runner.py` already includes a `__main__` block:

```python
if __name__ == "__main__":
    debug(
        algo_dir=os.path.dirname(__file__),
        data="path to data",
        params={"scale": 2.0, "bias": 1.0},
        device="cpu",
    )
```

Update `data` and `params`, then run:

```bash
python Runner.py
```

This is useful when you want to use `breakpoint()` directly in your IDE.

### Option 2: Debug through the CLI

You can also run the same algorithm through the Compox debug CLI:

```bash
compox debug run --algo "path to algorithm_template" --data "path to data" --device "cpu" --param scale=2.0 --param bias=1.0
```

`--param` can be repeated multiple times. JSON values are also supported for
structured parameters.

## Deployment

Deploy only this template with:

```bash
compox deploy-algorithms --config app_server.yaml --name algorithm_template
```

See [`../README.md`](../README.md) for the general algorithm contract and
typed runner overview.
