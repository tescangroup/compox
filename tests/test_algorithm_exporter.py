"""
Copyright 2026 Tescan GROUP, a.s.
All rights reserved
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from compox.algorithm_utils.AlgorithmExporter import AlgorithmExporter
from compox.database_connection.InMemoryConnection import InMemoryConnection
from compox.exceptions import CompoxNotFoundError


def _make_module_zip_bytes(module_id: str) -> bytes:
    """
    Build a module zip that matches AlgorithmDeployer output shape:
    zip contains a top-level folder so _extract_module_zip keeps contents.
    """
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            f"{module_id}/package/Runner.py",
            f"MODULE_ID = '{module_id}'\nclass Runner: pass\n",
        )
        zf.writestr(f"{module_id}/package/utils.py", "x = 1\n")
    buf.seek(0)
    return buf.read()


def _base_algorithm_record(
    algorithm_id: str,
    name: str,
    major: str,
    latest_minor: str,
    minor_versions: dict[str, dict],
) -> dict:
    return {
        "algorithm_id": algorithm_id,
        "algorithm_name": name,
        "algorithm_major_version": major,
        "latest_algorithm_minor_version": latest_minor,
        "algorithm_minor_version": minor_versions,
    }


@pytest.fixture
def db_with_algorithm():
    algorithm_id = "alg-1"
    name = "my_algo"
    major = "1"
    module_id = "module-abc"
    assets = {"weights.bin": "asset-a"}
    algorithm_record = _base_algorithm_record(
        algorithm_id=algorithm_id,
        name=name,
        major=major,
        latest_minor="0",
        minor_versions={
            "0": {
                "timestamp": "2026-01-01 00:00:00",
                "module_id": module_id,
                "assets": assets,
            }
        },
    )
    db = InMemoryConnection()

    # Match S3-style list_objects shape expected by AlgorithmExporter.
    db.list_objects = lambda collection_name: [
        {"Key": key} for key in db.store.get(collection_name, {}).keys()
    ]

    db.put_objects(
        "algorithm-store",
        [f"{algorithm_id}~{name}~{major}"],
        [json.dumps(algorithm_record)],
    )
    db.put_objects(
        "module-store",
        [module_id],
        [_make_module_zip_bytes(module_id)],
    )
    db.put_objects("asset-store", ["asset-a"], [b"AAA"])
    return db


def _read_zip_file(path: Path, relpath: str) -> bytes:
    """Read a single file from a zip archive by relative path."""
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open(relpath) as f:
            return f.read()


def test_export_algorithm_to_zip_uses_latest_minor(db_with_algorithm, tmp_path):
    """Exports the latest minor version and restores module + assets."""
    exporter = AlgorithmExporter(db_with_algorithm)
    target = tmp_path / "algo.zip"
    exporter.export_algorithm_to_zip(
        algorithm_name="my_algo",
        algorithm_major_version="1",
        target_zip_path=str(target),
    )

    # module files should be present
    assert _read_zip_file(target, "package/Runner.py").startswith(
        b"MODULE_ID"
    ), "Expected Runner.py to exist and include MODULE_ID header."
    # assets should be restored
    assert (
        _read_zip_file(target, "weights.bin") == b"AAA"
    ), "Expected asset 'weights.bin' to match default asset bytes for latest minor."


def test_export_algorithm_to_zip_specific_minor(db_with_algorithm, tmp_path):
    """Exports a requested minor version and validates module selection."""
    # add minor version 1
    algo_key = "alg-1~my_algo~1"
    algo_json = json.loads(
        db_with_algorithm.get_objects("algorithm-store", [algo_key])[0]
    )
    algo_json["algorithm_minor_version"]["1"] = {
        "timestamp": "2026-01-02 00:00:00",
        "module_id": "module-xyz",
        "assets": {"weights.bin": "asset-b"},
    }
    algo_json["latest_algorithm_minor_version"] = "1"
    db_with_algorithm.put_objects(
        "algorithm-store", [algo_key], [json.dumps(algo_json)]
    )
    db_with_algorithm.put_objects(
        "module-store", ["module-xyz"], [_make_module_zip_bytes("module-xyz")]
    )
    db_with_algorithm.put_objects("asset-store", ["asset-b"], [b"BBB"])

    exporter = AlgorithmExporter(db_with_algorithm)
    target = tmp_path / "algo_minor_0.zip"
    exporter.export_algorithm_to_zip(
        algorithm_name="my_algo",
        algorithm_major_version="1",
        algorithm_minor_version="0",
        target_zip_path=str(target),
    )

    assert (
        _read_zip_file(target, "weights.bin") == b"AAA"
    ), "Expected minor version 0 to export asset bytes from asset-a."
    runner_contents = _read_zip_file(target, "package/Runner.py").decode(
        "utf-8"
    )
    assert (
        "MODULE_ID = 'module-abc'" in runner_contents
    ), "Expected minor version 0 to include module-abc in Runner.py."

    target_v1 = tmp_path / "algo_minor_1.zip"
    exporter.export_algorithm_to_zip(
        algorithm_name="my_algo",
        algorithm_major_version="1",
        algorithm_minor_version="1",
        target_zip_path=str(target_v1),
    )
    runner_contents_v1 = _read_zip_file(target_v1, "package/Runner.py").decode(
        "utf-8"
    )
    assert (
        "MODULE_ID = 'module-xyz'" in runner_contents_v1
    ), "Expected minor version 1 to include module-xyz in Runner.py."


def test_export_algorithm_to_zip_checkpoint_override(
    db_with_algorithm, tmp_path
):
    """Overrides default assets with checkpoint assets during export."""
    # add checkpoint asset override
    checkpoint_manifest = {
        "checkpoint_id": "chk-1",
        "parent_algorithm_id": "alg-1",
        "training_id": "train-1",
        "assets": {"weights.bin": "asset-b"},
        "created_at": "2026-01-02T00:00:00Z",
        "properties": {},
        "tags": [],
        "parent_checkpoint_id": None,
    }
    db_with_algorithm.put_objects(
        "algorithm-checkpoint-store",
        ["chk-1"],
        [json.dumps(checkpoint_manifest)],
    )
    db_with_algorithm.put_objects("asset-store", ["asset-b"], [b"BBB"])

    exporter = AlgorithmExporter(db_with_algorithm)
    target = tmp_path / "algo_chk.zip"
    exporter.export_algorithm_to_zip(
        algorithm_name="my_algo",
        algorithm_major_version="1",
        target_zip_path=str(target),
        algorithm_checkpoint_id="chk-1",
    )

    assert (
        _read_zip_file(target, "weights.bin") == b"BBB"
    ), "Expected checkpoint asset override to replace weights.bin content."


def test_export_algorithm_zip_stream(db_with_algorithm):
    """Returns a filename and a byte stream for the exported zip."""
    exporter = AlgorithmExporter(db_with_algorithm)
    filename, tmp_path, stream = exporter.export_algorithm_zip_stream(
        algorithm_name="my_algo", algorithm_major_version="1"
    )
    assert (
        filename == "my_algo_v1"
    ), "Expected stream filename to follow <name>_v<major> format."
    # stream yields bytes
    first = next(iter(stream))
    assert isinstance(
        first, (bytes, bytearray)
    ), "Expected export stream to yield byte chunks."


def test_export_algorithm_errors(db_with_algorithm, tmp_path):
    """Raises the right errors for missing algorithm/minor/checkpoint."""
    exporter = AlgorithmExporter(db_with_algorithm)

    with pytest.raises(CompoxNotFoundError) as missing_algorithm_exc:
        exporter.export_algorithm_to_zip(
            algorithm_name="missing",
            algorithm_major_version="1",
            target_zip_path=str(tmp_path / "missing.zip"),
        )
    assert missing_algorithm_exc.value.code == "algorithm_not_found"

    with pytest.raises(CompoxNotFoundError) as missing_minor_exc:
        exporter.export_algorithm_to_zip(
            algorithm_name="my_algo",
            algorithm_major_version="1",
            algorithm_minor_version="99",
            target_zip_path=str(tmp_path / "missing_minor.zip"),
        )
    assert missing_minor_exc.value.code == "minor_version_not_found"

    with pytest.raises(CompoxNotFoundError) as missing_checkpoint_exc:
        exporter.export_algorithm_to_zip(
            algorithm_name="my_algo",
            algorithm_major_version="1",
            target_zip_path=str(tmp_path / "missing_chk.zip"),
            algorithm_checkpoint_id="nope",
        )
    assert missing_checkpoint_exc.value.code == "checkpoint_not_found"
