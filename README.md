# Compox
Compox is a simple Python execution engine for Tescan applications. It's purpose is to run data processing algorithms written in Python on the server and report the results to the client application such as Tescan 3D Viewer or Picannto.

See the [GitHub pages](https://tescangroup.github.io/compox) for full python API reference, docs and tutorials!

## Installation
Compox can be installed through PyPI: 
```bash
pip install compox
```

You can also install the server by cloning this repository and installing it from source.

Note that some additional dependencies, such as MinIO executable for your operating system, will be downloaded on the first run of the server.

## Running the server
Once installed into your Python environment, Compox can be run using the `compox` command. This command is available in the virtual environment 
created during the setup process. Run the following command to list the available commands:

```bash
compox --help
```

It's recommended to start with generating an initial configuration file with `compox generate-config`. See the configuration options section below.
To see the usage of the `generate-config` command, you can run:

```bash
compox generate-config --help
```

To run the Compox, use the following command:

```bash
compox run --config /path/to/config.yaml
```

### Deploying algorithms
See the relevant section in the [documentation](https://tescangroup.github.io/compox/development.html#how-to-create-an-algorithm-module) for information about target structure of the deployable algorithms. You may also refer to our [tutorials](https://tescangroup.github.io/compox/tutorials.html) that will guide you through the preparation and deployment of selected algorithms.

Once your algorithm is ready, you can use the `compox deploy-algorithms` command to deploy it to your server. 
To see the usage of the `deploy-algorithms` command, you can run:

```bash
compox deploy-algorithms --help
```

By default, this command reads algorithm definitions from the folder specified
in your configuration file with the `deploy_algorithms_from` key and deploys
all deployable algorithm subfolders found there.

Examples:

```bash
compox deploy-algorithms --config /path/to/config.yaml
```

Deploy only selected algorithm folders:

```bash
compox deploy-algorithms --config /path/to/config.yaml --name foo --name bar
```

Override the algorithm root folder for a single run:

```bash
compox deploy-algorithms --config /path/to/config.yaml --path /path/to/algorithms
```

Note that the server does not need to be running in order to deploy the algorithms.

### Algorithm Bundles
Compox can package builtin algorithms into an encrypted algorithm bundle and later
import that bundle into another backend. This is useful when you want to ship a
predefined set of builtin algorithms together with their code, assets and checkpoint
metadata.

An algorithm bundle is a snapshot of these storage collections:

- `algorithm-store`
- `module-store`
- `asset-store`
- `algorithm-checkpoint-store`

Build a bundle from the currently configured storage backend:

```bash
compox build-algorithm-bundle --config /path/to/config.yaml
```

If you do not provide a key explicitly, Compox generates one and prints it. You can
also generate a key separately:

```bash
compox generate-algorithm-bundle-key
```

Inspect a bundle without importing it:

```bash
compox inspect-algorithm-bundle \
  --bundle-path /path/to/compox_algorithm_bundle.zip \
  --key <bundle-key>
```

Import a bundle into the backend configured by your config file:

```bash
compox import-algorithm-bundle \
  --config /path/to/config.yaml \
  --bundle-path /path/to/compox_algorithm_bundle.zip \
  --key <bundle-key>
```

If `storage.builtin_storage_bundle_path` and `storage.builtin_storage_bundle_key`
are configured, Compox also imports the bundle automatically during server startup.
The startup import is gated by the bundle file hash, so the same bundle is not
re-imported repeatedly after a successful import.

Bundle import is semantic, and follows the AlgorithmDeployer logic with some special handling for bundled algorithms:

- if a builtin algorithm with the same name and major version already exists,
  Compox reuses the existing algorithm id and applies the standard minor-version
  insertion rule used by normal algorithm deployment
- if the bundled latest minor version has the same `module_id` and `assets` as
  the current latest target version, no new minor version is created
- if the bundled latest minor version differs, a new minor version is appended
- checkpoint manifests are imported and their `parent_algorithm_id` is rewritten
  to the target algorithm id

If an algorithm with the same name and major version already exists in the target
backend, Compox treats it as the merge target and applies the same minor-version
update logic. The import does not use a separate ownership flag.


## Configuration
The server uses pydantic settings for configuration. The options can be either set as runtime arguments, in a yaml file, or using the `COMPOX__` environment variables, in corresponding order of precedence.

- Ensure all paths and URLs are correctly set before running the application.
- Adjust CUDA settings based on hardware capabilities.
- Logging paths should be accessible by the application to prevent errors.
- Use the `logging` section to control how much detail is shown in the console versus the log file.
- See the [configuration reference](#configuration-reference) below for detailed configuration options.

<details><summary>Configuration reference</summary>

<br>

### Compox Configuration Reference

| Section                           | Field                        | Default                              | Description                                                                 |
|------------------------------------|------------------------------|--------------------------------------|-----------------------------------------------------------------------------|
|                                    | `port`                       | `5481`                               | The main server port used to make requests.                                 |
|                                    | `deploy_algorithms_from`     | `"./algorithms"`                     | Directory for algorithm deployment sources.                                 |
|                                    | `log_path`                   | `"LOG_DEFAULT:compox.log"`           | Path to the main log file (supports dynamic prefixes).                      |
|                                    | `config`                     | `None`                               | Optional config path override.                                              |
| `logging`                          | `console_level`              | `"INFO"`                             | Minimum level shown in the interactive console. At `DEBUG`/`TRACE`, suppressed access and MinIO console logs are shown again. |
| `logging`                          | `file_level`                 | `"INFO"`                             | Minimum level written to the main log file.                                 |
| `info`                             | `product_name`               | `"Tescan Compox Backend"`                | Product display name.                                                       |
| `info`                             | `server_tags`                | `[]` (auto appends `"compox"`)       | Tags attached to the server. `"compox"` is added automatically.             |
| `info`                             | `group_name`                 | `"TESCAN GROUP, a.s."`               | Name of the corporate group.                                                |
| `info`                             | `organization_name`          | `"TESCAN GROUP, a.s."`              | Full name of the organization.                                              |
| `info`                             | `organization_domain`        | `"tescan.com"`                   | Domain used in server configuration.                                        |
| `gui`                              | `algorithm_add_remove_in_menus` | `False`                           | Enables/disables GUI menu for algorithm management.                         |
| `gui`                              | `use_systray`                | `False`                              | Enables/disables systray GUI integration.                                   |
| `gui`                              | `icon_path`                  | Path to the installed package resource | Path to the systray icon (supports dynamic prefixes).                    |
| `inference`                        | `device`                     | `"cuda"`                             | Device used for model inference (`"cpu"`, `"cuda"`, `"mps"`).               |
| `inference`                        | `cuda_visible_devices`       | `"0"`                                | Comma-separated list of visible CUDA GPUs.                                  |
| `inference`                        | `algorithm_cache_maxsize`    | `1`                                  | Maximum number of cached algorithm runner variants kept per process.        |
| `inference.backend_settings` (fastapi) | `executor`               | `"fastapi_background_tasks"`         | Task executor selection.                                                    |
| `inference.backend_settings` (fastapi) | `worker_number`          | `1`                                  | Number of FastAPI background task workers.                                  |
| `inference.backend_settings` (celery) | `executor`                | `"celery"`                           | Task executor selection.                                                    |
| `inference.backend_settings` (celery) | `worker_name`            | `"compox_worker"`                    | Celery worker name.                                                         |
| `inference.backend_settings` (celery) | `broker_url`             | required                             | Broker URL for Celery (e.g. `amqp://`).                                     |
| `inference.backend_settings` (celery) | `result_backend`         | `"rpc://"`                           | Celery result backend.                                                      |
| `inference.backend_settings` (celery) | `run_flower`             | `False`                              | Whether to run Celery Flower.                                               |
| `inference.backend_settings` (celery) | `flower_port`            | `None`                               | Optional Flower port override.                                              |
| `storage`                          | `collection_prefix`          | `""`                                 | Prefix applied to object store collections (useful for AWS S3).             |
| `storage`                          | `data_store_expire_days`     | `1`                                  | Days until stored datasets expire.                                          |
| `storage`                          | `execution_store_expire_days`| `30`                                 | Days until execution data expires.                                          |
| `storage`                          | `training_store_expire_days` | `30`                                 | Days until training data expires.                                           |
| `storage`                          | `deploy_store_expire_days`   | `30`                                 | Days until deploy data expires.                                             |
| `storage`                          | `stop_requests_expire_days`  | `7`                                  | Days until stop-requests expire.                                            |
| `storage`                          | `access_key_id`              | generated with `UUIDv4`              | Generated access key for storage backend. If `null` is provided, random UUIDv4 is generated. |
| `storage`                          | `secret_access_key`          | generated with `UUIDv4`              | Generated secret key for storage backend. If `null` is provided, random UUIDv4 is generated. |
| `storage`                          | `builtin_storage_bundle_path`| `None`                               | Optional path to an encrypted builtin algorithm bundle imported at startup.  |
| `storage`                          | `builtin_storage_bundle_key` | `None`                               | Base64url-encoded AES-256 key used to read the builtin algorithm bundle.     |
| `storage.backend_settings` (minio) | `provider`                   | `"minio"`                            | Selected backend provider.                                                  |
| `storage.backend_settings` (minio) | `start_instance`             | `True`                               | Whether to start a local MinIO server.                                      |
| `storage.backend_settings` (minio) | `port`                       | `5483`                               | MinIO service port.                                                         |
| `storage.backend_settings` (minio) | `console_port`               | `5482`                               | MinIO admin console port.                                                   |
| `storage.backend_settings` (minio) | `executable_path`            | `"minio/minio_bin"` or `"minio/minio.exe"` | MinIO binary path (OS-specific, supports dynamic prefixes).            |
| `storage.backend_settings` (minio) | `storage_path`               | `"minio/compox_store"`               | Storage directory used by MinIO (supports dynamic prefixes).                |
| `storage.backend_settings` (minio) | `aws_region`                 | `None`                               | Optional AWS compatibility region.                                          |
| `storage.backend_settings` (minio) | `s3_domain_name`             | `None`                               | Optional domain override for S3 compatibility.                              |
| `storage.backend_settings` (minio) | `s3_endpoint_url`            | Derived from `port`                  | Computed as `http://localhost:{port}`.                                      |
| `storage.backend_settings` (aws)   | `provider`                   | `"aws"`                              | AWS backend selection.                                                      |
| `storage.backend_settings` (aws)   | `s3_endpoint_url`            | `None`                               | Optional override for S3 endpoint URL.                                      |
| `storage.backend_settings` (aws)   | `aws_region`                 | `None`                               | AWS region (e.g. `us-east-1`).                                              |
| `storage.backend_settings` (aws)   | `s3_domain_name`             | `None`                               | Domain used for S3-style URLs.                                              |
| `ssl`                              | `use_ssl`                    | `False`                              | Enable SSL on the server.                                                   |
| `ssl`                              | `ssl_keyfile`                | `None`                               | SSL key file path (supports dynamic prefixes).                              |
| `ssl`                              | `ssl_certfile`               | `None`                               | SSL certificate path (supports dynamic prefixes).                           |
| `middleware`                       | `allow_origins`              | `[]`                                 | CORS allowed origins.                                                       |
| `middleware`                       | `allow_methods`              | `["GET"]`                            | CORS allowed methods.                                                       |
| `middleware`                       | `allow_headers`              | `[]`                                 | CORS allowed headers.                                                       |
| `middleware`                       | `allow_credentials`          | `False`                              | CORS allow credentials.                                                     |
| `middleware`                       | `expose_headers`             | `[]`                                 | CORS exposed headers.                                                       |
| `middleware`                       | `max_age`                    | `3600`                               | CORS preflight max age in seconds.                                          |

### Logging configuration

Compox uses separate console and file sinks. This lets you keep the terminal readable while still collecting more detailed logs in the file.

Example:

```yaml
log_path: "LOG_DEFAULT:compox.log"

logging:
  console_level: "INFO"
  file_level: "DEBUG"
```

Recommended modes:

- Quiet day-to-day operation:
  ```yaml
  logging:
    console_level: "INFO"
    file_level: "DEBUG"
  ```
- Full interactive debugging:
  ```yaml
  logging:
    console_level: "DEBUG"
    file_level: "DEBUG"
  ```

At `INFO` console level, Compox suppresses some high-frequency console noise such as polling status requests, file transfer access logs, and routine MinIO subprocess output. These logs still remain available in the file sink. At `DEBUG` or `TRACE`, those console logs are shown again.

### Emergency fallback records

Compox keeps a small filesystem side store for terminal fallback records when the
primary storage backend cannot persist the final execution, training, or deploy
state.

Behavior:

- Fallback records are written only for:
  - `execution-store`
  - `training-store`
  - `deploy-store`
- The emergency store root defaults to:
  - `<directory of log_path>/compox_emergency_records`
  - or the system temp directory if `log_path` is not available
- Fallback files use a Compox-specific filename pattern:
  - `compox_emergency_<record_id>.json`
- Existing fallback files are purged on Compox startup for the shared app/worker
  emergency store
- The startup purge only removes Compox-owned fallback files in the known
  fallback collections above; it does not delete arbitrary files under the root
- If a later write of the same terminal record succeeds in the primary storage,
  the matching emergency fallback record is deleted

The emergency store is intentionally a best-effort same-run recovery mechanism,
not a long-term archival store.

### Dynamic path prefixes

Some fields in the Compox configuration (such as `log_path`, `icon_path`, etc.) support **dynamic prefixes** that resolve to OS-specific or runtime-specific paths. This allows for portability across platforms (e.g., Windows, Linux) and between development and production environments.

#### Supported Prefixes

| Prefix                   | Meaning (Resolved To...)                                                                 |
|--------------------------|------------------------------------------------------------------------------------------|
| `LOG_DEFAULT:`           | A platform-dependent log directory:                                                     |
|                          | - Windows: `%TEMP%/<organization>/<product>`                                            |
|                          | - Linux/macOS: `/var/log/<organization>/<product>`                                      |
| `PROGRAMDATA_DEFAULT:`   | A system-wide data directory (Windows only):                                            |
|                          | - e.g., `%PROGRAMDATA%/<organization>/<product>`                                        |
|                          | - On Linux/macOS, defaults to `"."` (current dir)                                       |

These are resolved **at runtime** in the `Settings.parse_paths()` validator method.

## Security Notice
While you can technically communicate with a remotely running Compox instance, the communication is currently not authenticated and done via standard HTTP. Do not expose Compox endpoints to network in case of sensitive code or data. Security improvements are currently work in progress.
