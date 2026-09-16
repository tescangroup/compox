"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import toml
import pytest
import json
import os
import py_compile
from unittest.mock import call
import hashlib
import shutil

from compox.algorithm_utils.AlgorithmDeployer import AlgorithmDeployer


@pytest.fixture
def valid_alg_dir(tmp_path):
    """
    Create a temporary algorithm directory with a valid pyproject.toml and Runner.py.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Base temporary directory fixture provided by pytest.

    Returns
    -------
    str
        Path to the created algorithm directory containing:
          - pyproject.toml with valid data
          - Runner.py file
    """
    directory = tmp_path / "alg"
    directory.mkdir()
    content = {
        "project": {"name": "my_algo", "version": "1.2"},
        "tool": {
            "compox": {
                "check_importable": False,
                "obfuscate": False,
                "algorithm_type": "Generic",
                "tags": ["denoising"],
                "description": "denoising algorithm",
                "supported_devices": ["cpu"],
                "default_device": "cpu",
                "additional_parameters": [],
            }
        },
    }

    toml_path = directory / "pyproject.toml"
    toml_path.write_text(toml.dumps(content))
    (directory / "Runner.py").write_text("class Runner: pass")
    return str(directory)


@pytest.fixture
def invalid_alg_dir(tmp_path):
    """
    Create a temporary algorithm directory with an invalid pyproject.toml and Runner.py.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Base temporary directory fixture provided by pytest.

    Returns
    -------
    str
        Path to the created algorithm directory containing:
          - pyproject.toml with invalid data
          - Runner.py file
    """
    d = tmp_path / "invalid"
    d.mkdir()
    (d / "pyproject.toml").write_text(
        toml.dumps({"some_invalid_section": {"foo": "bar"}})
    )
    (d / "Runner.py").write_text("class Runner: pass")
    return str(d)


# Test 1 - parse valid pyproject
def test_parse_pyproject(valid_alg_dir):
    """
    Verify that AlgorithmDeployer extracts the correct metadata from pyproject.toml.
    """
    deployer = AlgorithmDeployer(valid_alg_dir)

    assert (
        deployer.algorithm_name == "my_algo"
    ), f"Expected algorithm_name to be 'my_algo', got {deployer.algorithm_name!r}"
    assert (
        deployer.algorithm_major_version == "1"
    ), f"Expected algorithm_major_version to be '1', got {deployer.algorithm_major_version!r}"
    assert (
        deployer.algorithm_minor_version == "2"
    ), f"Expected algorithm_minor_version to be '2', got {deployer.algorithm_minor_version!r}"
    assert (
        deployer.algorithm_type == "Generic"
    ), f"Expected algorithm_type to be 'Generic', got {deployer.algorithm_type!r}"


# Test 2 - parse invalid pyproject
@pytest.mark.filterwarnings("ignore:.*Algorithm type is set to.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*description.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*tags.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*supported_devices.*:UserWarning")
def test_parse_invalid_pyproject(invalid_alg_dir):
    """
    Verify that AlgorithmDeployer raises Exception for invalid input.
    """
    with pytest.raises(KeyError):
        AlgorithmDeployer(invalid_alg_dir)

    with pytest.raises(FileNotFoundError):
        AlgorithmDeployer("invalid_path")


# Test 3 - find py files
def test_find_py_files(valid_alg_dir):
    """
    Verify that find_py_files returns all .py files.
    """

    deployer = AlgorithmDeployer(valid_alg_dir)
    files, hashes = deployer.find_py_files(valid_alg_dir)
    assert any(
        f.endswith(".py") for f in files
    ), f"Expected to find 'Runner.py' in the list of Python files, got {files!r}"


def test_find_py_files_ignores_git_metadata(tmp_path):
    """
    Verify that find_py_files ignores Python files stored under .git metadata.
    """
    d = tmp_path / "dir"
    d.mkdir()
    (d / "Runner.py").write_text("class Runner:\n    pass\n")
    git_dir = d / ".git"
    git_dir.mkdir()
    (git_dir / "hook.py").write_text("print('do not include')\n")

    files, _ = AlgorithmDeployer.find_py_files(str(d))

    assert any(
        f.endswith("Runner.py") for f in files
    ), f"Expected 'Runner.py' in output files, got {files!r}"
    assert all(
        ".git" not in f.split(os.path.sep) for f in files
    ), f"Did not expect files in '.git' directories, got {files!r}"


# Test 4 - store algorithm with module-store, asset-store and algorithm-store
def test_store_algorithm(valid_alg_dir, mock_connection):
    """
    Verify that `store_algorithm` writes module, assets, and algorithm metadata
    to the mock database in the correct order and format.
    """
    mock_connection.list_collections.return_value = []
    deployer = AlgorithmDeployer(valid_alg_dir)
    returned_id = deployer.store_algorithm(database_connection=mock_connection)
    calls = mock_connection.put_objects.call_args_list

    expected_calls = [
        call(["module-store"]),
        call(["asset-store"]),
        call(["algorithm-store"]),
    ]
    assert mock_connection.create_collections.call_count == 3, (
        f"Expected create_collections to be called 3 times (module-store, asset-store, algorithm-store)"
        f"but was called {mock_connection.create_collections.call_count} times"
    )
    assert mock_connection.put_objects.call_count == 2, (
        f"Expected put_objects to be called 3 times (module-store, asset-store, algorithm-store)"
        f"but was called {mock_connection.put_objects.call_count} times"
    )
    assert (
        mock_connection.create_collections.call_args_list == expected_calls
    ), (
        f"Expected call_args_list to be {expected_calls!r},"
        f"got { mock_connection.create_collections.call_args_list!r}"
    )

    # 3) algorithm-store, and verify its key + payload
    _, keys3, values3 = calls[1][0]
    payload = json.loads(values3[0])
    expected_key = (
        f"{returned_id}~{deployer.algorithm_name}~"
        f"{deployer.algorithm_major_version}"
    )
    assert keys3 == [
        expected_key
    ], f"Expected algorithm-store key to be {expected_key!r}, got {keys3!r}"
    assert (
        payload["algorithm_id"] == returned_id
    ), f"Expected payload['algorithm_id'] == {returned_id!r}, got {payload['algorithm_id']!r}"
    assert (
        payload["algorithm_name"] == "my_algo"
    ), f"Expected payload['algorithm_name'] == 'my_algo', got {payload['algorithm_name']!r}"
    assert (
        payload["algorithm_major_version"] == "1"
    ), f"Expected payload['algorithm_major_version'] == '1', got {payload['algorithm_major_version']!r}"
    assert payload.get(
        "timestamp"
    ), f"Expected payload to contain a non-empty 'timestamp', got {payload.get('timestamp')!r}"


# Test 5 - store algorithm and skip algorithm-store
def test_store_algorithm_skips_only_algorithm_store(
    valid_alg_dir, mock_connection
):
    """
    Verify that `store_algorithm` skips creating the "algorithm-store" collection
    when it already exists, but still uploads module and asset data.
    """
    mock_connection.list_collections.return_value = ["algorithm-store"]
    deployer = AlgorithmDeployer(valid_alg_dir)
    deployer.store_algorithm(database_connection=mock_connection)
    created = [
        args[0] for args, _ in mock_connection.create_collections.call_args_list
    ]

    assert mock_connection.create_collections.call_count == 2, (
        f"Expected create_collections to be called 2 times (module-store, asset-store)"
        f"but was called {mock_connection.create_collections.call_count} times"
    )
    assert [
        "module-store"
    ] in created, (
        f"Expected 'module-store' in created connection, got {created!r}"
    )
    assert [
        "asset-store"
    ] in created, (
        f"Expected 'asset-store' in created connection, got {created!r}"
    )
    assert all(
        c != ["algorithm-store"] for c in created
    ), f"Did not expect 'algorithm-store' in created connection, got {created!r}"


# Test 6 - parse pyproject_toml
def test_parse_pyproject_toml_direct(valid_alg_dir):
    """
    Verify that parse_pyproject_toml correctly reads pyproject.toml file and returns dict.
    """
    deployer = AlgorithmDeployer(valid_alg_dir)
    parsed = deployer.parse_pyproject_toml(valid_alg_dir)
    assert isinstance(
        parsed, dict
    ), f"'parse_pyproject_toml' should return 'dict', got {type(parsed)!r}"
    assert (
        parsed["project"]["name"] == "my_algo"
    ), f"Expected parsed data 'name' to be 'my_algo', got {parsed['project']['name']!r}"
    assert (
        parsed["project"]["version"] == "1.2"
    ), f"Expected parsed data 'version' to be '1.2', got {parsed['project']['version']!r}"


# Test 7 - apostrophe handling in additional parameter descriptions
def test_parse_pyproject_additional_parameters_with_apostrophe(tmp_path):
    """
    Valid TOML using double-quoted strings should preserve apostrophes and all
    additional parameters.
    """
    d = tmp_path / "apostrophe_valid"
    d.mkdir()
    (d / "Runner.py").write_text("class Runner: pass")
    (d / "pyproject.toml").write_text(
        """
[project]
name = "apostrophe_algo"
version = "1.2"

[tool.compox]
algorithm_type = "Generic"
tags = ["test"]
description = "apostrophe handling"
supported_devices = ["cpu"]
default_device = "cpu"
additional_parameters = [
  { name = "first", description = "User's first parameter", config = { type = "string", default = "a" } },
  { name = "second", description = "Second parameter", config = { type = "string", default = "b" } }
]
""".strip()
    )

    deployer = AlgorithmDeployer(str(d))

    assert len(deployer.additional_parameters) == 2
    assert (
        deployer.additional_parameters[0]["description"]
        == "User's first parameter"
    )
    assert deployer.additional_parameters[1]["name"] == "second"


# Test 8 - displayed_name defaults to a client-friendly label
def test_additional_parameter_displayed_name_defaults_from_name(tmp_path):
    d = tmp_path / "displayed_name_default"
    d.mkdir()
    (d / "Runner.py").write_text("class Runner: pass")
    (d / "pyproject.toml").write_text(
        """
[project]
name = "display_name_algo"
version = "1.2"

[tool.compox]
algorithm_type = "Generic"
tags = ["test"]
description = "displayed_name handling"
supported_devices = ["cpu"]
default_device = "cpu"
additional_parameters = [
  { name = "segmentation_threshold", description = "Threshold used for segmentation.", config = { type = "float", default = 0.5 } }
]
""".strip()
    )

    deployer = AlgorithmDeployer(str(d))

    assert (
        deployer.additional_parameters[0]["displayed_name"]
        == "Segmentation threshold"
    )


# Test 9 - explicit displayed_name should be preserved
def test_additional_parameter_displayed_name_preserved(tmp_path):
    d = tmp_path / "displayed_name_explicit"
    d.mkdir()
    (d / "Runner.py").write_text("class Runner: pass")
    (d / "pyproject.toml").write_text(
        """
[project]
name = "display_name_algo"
version = "1.2"

[tool.compox]
algorithm_type = "Generic"
tags = ["test"]
description = "displayed_name handling"
supported_devices = ["cpu"]
default_device = "cpu"
additional_parameters = [
  { name = "segmentation_threshold", displayed_name = "Segmentation threshold", description = "Threshold used for segmentation.", config = { type = "float", default = 0.5 } }
]
""".strip()
    )

    deployer = AlgorithmDeployer(str(d))

    assert (
        deployer.additional_parameters[0]["displayed_name"]
        == "Segmentation threshold"
    )


# Test 10 - decimal precision is preserved for float parameters
def test_float_parameter_decimal_precision_preserved(tmp_path):
    d = tmp_path / "decimal_precision_valid"
    d.mkdir()
    (d / "Runner.py").write_text("class Runner: pass")
    (d / "pyproject.toml").write_text(
        """
[project]
name = "decimal_precision_algo"
version = "1.2"

[tool.compox]
algorithm_type = "Generic"
tags = ["test"]
description = "decimal precision handling"
supported_devices = ["cpu"]
default_device = "cpu"
additional_parameters = [
  { name = "segmentation_threshold", description = "Threshold used for segmentation.", config = { type = "float", default = 0.5, decimal_precision = 3 } }
]
""".strip()
    )

    deployer = AlgorithmDeployer(str(d))

    assert deployer.additional_parameters[0]["config"]["decimal_precision"] == 3


# Test 11 - decimal precision should be rejected for non-float parameters
def test_non_float_parameter_decimal_precision_raises_value_error(tmp_path):
    d = tmp_path / "decimal_precision_invalid"
    d.mkdir()
    (d / "Runner.py").write_text("class Runner: pass")
    (d / "pyproject.toml").write_text(
        """
[project]
name = "decimal_precision_algo"
version = "1.2"

[tool.compox]
algorithm_type = "Generic"
tags = ["test"]
description = "decimal precision handling"
supported_devices = ["cpu"]
default_device = "cpu"
additional_parameters = [
  { name = "iterations", description = "Iteration count.", config = { type = "int", default = 5, decimal_precision = 2 } }
]
""".strip()
    )

    with pytest.raises(ValueError, match="decimal_precision"):
        AlgorithmDeployer(str(d))


# Test 12 - malformed apostrophe quoting should fail clearly
def test_parse_pyproject_invalid_apostrophe_quoting_raises_value_error(
    tmp_path,
):
    """
    Invalid TOML should raise a clear error instead of degrading into partial
    additional parameter parsing.
    """
    d = tmp_path / "apostrophe_invalid"
    d.mkdir()
    (d / "Runner.py").write_text("class Runner: pass")
    (d / "pyproject.toml").write_text(
        """
[project]
name = "apostrophe_algo"
version = "1.2"

[tool.compox]
algorithm_type = "Generic"
tags = ["test"]
description = "apostrophe handling"
supported_devices = ["cpu"]
default_device = "cpu"
additional_parameters = [
  { name = "first", description = 'User's first parameter', config = { type = "string", default = "a" } },
  { name = "second", description = "Second parameter", config = { type = "string", default = "b" } }
]
""".strip()
    )

    with pytest.raises(ValueError, match="Invalid pyproject.toml"):
        AlgorithmDeployer(str(d))


# Test 13 - process path to dict
def test_process_path_to_dict_key():
    """
    Verify that process_path_to_dict_key:
        - converts '\\' and '\' to '/'
        - delete leading slashes or backslashes
    """
    raw = r"\foo\bar/baz"
    key = AlgorithmDeployer.process_path_to_dict_key(raw)
    assert (
        key == "foo/bar/baz"
    ), f" Expected 'key' to be 'foo/bar/baz', got {key!r}"


# Test 14 - find other than py files
def test_find_other_than_py_files(tmp_path):
    """
    Verify that find_other_than_py_files returns only non- '.py' files
    and exclude files in __pycache__ directory.
    """
    d = tmp_path / "dir"
    d.mkdir()
    (d / "a.txt").write_text("1")
    (d / "b.py").write_text("2")
    (d / "c.pyc").write_bytes(b"3")
    sub = d / "__pycache__"
    sub.mkdir()
    (sub / "d.txt").write_text("4")

    files = AlgorithmDeployer.find_other_than_py_files(str(d))
    assert any(
        f.endswith("a.txt") for f in files
    ), f"Expected 'a.txt' in output files, got {files!r}"
    assert all(
        not f.endswith(".py") for f in files
    ), f"Did not expect 'b.py' in output files, got {files!r}"
    assert all(
        not f.endswith(".pyc") for f in files
    ), f"Did not expect 'c.pyc' in output files, got {files!r}"
    assert all(
        "__pycache__" not in f for f in files
    ), f"Did not expect files in '__pycache__' directory, got {files!r}"


def test_create_algorithm_module_from_compiled_artifact(tmp_path):
    """
    Verify that bytecode-only algorithm directories are accepted as module input.
    """
    directory = tmp_path / "compiled_alg"
    directory.mkdir()
    content = {
        "project": {"name": "compiled_algo", "version": "1.2"},
        "tool": {
            "compox": {
                "check_importable": False,
                "obfuscate": False,
                "algorithm_type": "Generic",
                "tags": ["compiled"],
                "description": "compiled algorithm",
                "supported_devices": ["cpu"],
                "default_device": "cpu",
                "additional_parameters": [],
            }
        },
    }
    (directory / "pyproject.toml").write_text(toml.dumps(content))
    runner_py = directory / "Runner.py"
    runner_py.write_text("class Runner: pass\n")
    py_compile.compile(
        str(runner_py),
        cfile=str(directory / "Runner.pyc"),
        dfile="Runner.py",
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
    )
    runner_py.unlink()

    deployer = AlgorithmDeployer(str(directory))
    module_id, module_bytes = deployer._create_algorithm_module(
        str(directory), pyc_only=True
    )

    assert module_id, "Expected non-empty module id for compiled artifact."
    assert (
        module_bytes
    ), "Expected non-empty module bytes for compiled artifact."


def test_rename_module_path_retries_transient_permission_error(
    valid_alg_dir, tmp_path, monkeypatch
):
    """
    Verify transient file locks during module rename are retried.
    """
    source = tmp_path / "module_staging"
    target = tmp_path / "module_final"
    source.mkdir()
    (source / "Runner.py").write_text("class Runner: pass\n")

    original_rename = os.rename
    attempts = {"count": 0}

    def flaky_rename(src, dst):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("simulated transient file lock")
        original_rename(src, dst)

    monkeypatch.setattr(
        "compox.algorithm_utils.AlgorithmDeployer.os.rename", flaky_rename
    )
    monkeypatch.setattr(
        "compox.algorithm_utils.AlgorithmDeployer.time.sleep",
        lambda _seconds: None,
    )

    deployer = AlgorithmDeployer(valid_alg_dir)
    deployer._rename_module_path_with_retries(
        str(source),
        str(target),
        retries=5,
        delay_seconds=0,
    )

    assert attempts["count"] == 3
    assert target.exists()
    assert not source.exists()


def test_rename_module_path_does_not_retry_non_transient_error(
    valid_alg_dir, tmp_path, monkeypatch
):
    """
    Verify unrelated rename failures are not hidden by the retry helper.
    """
    source = tmp_path / "module_staging"
    target = tmp_path / "module_final"
    source.mkdir()
    attempts = {"count": 0}

    def failing_rename(_src, _dst):
        attempts["count"] += 1
        raise FileNotFoundError("simulated unrelated rename failure")

    monkeypatch.setattr(
        "compox.algorithm_utils.AlgorithmDeployer.os.rename", failing_rename
    )

    deployer = AlgorithmDeployer(valid_alg_dir)
    with pytest.raises(FileNotFoundError):
        deployer._rename_module_path_with_retries(str(source), str(target))

    assert attempts["count"] == 1


def test_find_other_than_py_files_ignores_git_metadata(tmp_path):
    """
    Verify that find_other_than_py_files ignores Git metadata files and directories.
    """
    d = tmp_path / "dir"
    d.mkdir()
    (d / "asset.bin").write_text("1")
    (d / ".gitignore").write_text("*.pyc\n")
    (d / ".gitmodules").write_text("[submodule]\n")
    git_dir = d / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

    files = AlgorithmDeployer.find_other_than_py_files(str(d))

    assert any(
        f.endswith("asset.bin") for f in files
    ), f"Expected 'asset.bin' in output files, got {files!r}"
    assert all(
        ".git" not in f.split(os.path.sep) for f in files
    ), f"Did not expect files in '.git' directories, got {files!r}"
    assert all(
        not f.endswith((".gitignore", ".gitmodules", ".gitattributes"))
        for f in files
    ), f"Did not expect Git metadata files in output, got {files!r}"


# Test 15 - calculate etag
def test_calculate_etag():
    """
    Verify that 'calculate_etag' returns an MD5-based etag
    """
    data = b"abc"
    etag = AlgorithmDeployer.calculate_etag(data)
    expected = '"' + hashlib.md5(data).hexdigest() + '"'
    assert etag == expected, f"Expected 'etag' to be {expected!r}, got {etag!r}"


# Test 16 - generate uuid with different versions
def test_generate_uuid_versions():
    """
    Verify that 'generate_uuid':
        - returns valid uuid for version 1 or 4
        - raise ValueError for any other versions
    """
    u1 = AlgorithmDeployer.generate_uuid(version=1)
    u4 = AlgorithmDeployer.generate_uuid(version=4)
    assert isinstance(u1, str) and len(u1) > 0, (
        f"'generate_uuid' should return 'str', got {type(u1)!r}. "
        f"Length of returned uuid string should be at least 1, got {len(u1)}"
    )
    assert isinstance(u4, str) and len(u4) > 0, (
        f"'generate_uuid' should return 'str', got {type(u4)!r}. "
        f"Length of returned uuid string should be at least 1, got {len(u4)}"
    )
    with pytest.raises(ValueError):
        AlgorithmDeployer.generate_uuid(version=2)


# Test 17 - check if zip is importable
@pytest.mark.filterwarnings(
    "ignore:.*zipimport.zipimporter.load_module.*:DeprecationWarning"
)
def test_check_if_zip_is_importable(tmp_path):
    """
    Verify that 'check_if_zip_is_importable' correctly intentifies importable and corrupt ZIP.
    """
    # good zip
    mod = tmp_path / "mod"
    mod.mkdir()
    (mod / "Runner.py").write_text("class Runner: pass")
    (mod / "__init__.py").write_text("")
    zip_path = shutil.make_archive(str(mod), "zip", str(mod))
    assert AlgorithmDeployer.check_if_zip_is_importable(zip_path, "Runner"), (
        f"'check_if_zip_is_importable' should return 'True' for valid ZIP, "
        f"got {AlgorithmDeployer.check_if_zip_is_importable(zip_path, 'Runner')!r}"
    )

    # bad zip
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    assert not AlgorithmDeployer.check_if_zip_is_importable(
        str(bad), "Runner"
    ), (
        f"'check_if_zip_is_importable' should return 'False' for invalid ZIP, "
        f"got {AlgorithmDeployer.check_if_zip_is_importable(str(bad), 'Runner')!r}"
    )


# Test 18 - minimalize py file
def test_minimize_py_files(valid_alg_dir, tmp_path):
    """
    Verify that `_minimize_py_files` delete comments and blank lines from Python files.
    """
    f = tmp_path / "m.py"
    f.write_text("# comment\n\nx = 1\n")
    deployer = AlgorithmDeployer(valid_alg_dir)
    deployer._minimize_py_files([str(f)])
    content = f.read_text()
    assert (
        "x=1" in content or "x = 1" in content
    ), f"Expected code 'x=1' or 'x = 1' in content after minimalization, got {content!r}"
    assert (
        "#" not in content
    ), f"Did not expect any '#' in content after minimalization, got {content!r}"
