"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

import json
import toml
import pytest

from compox.algorithm_utils.AlgorithmDeployer import AlgorithmDeployer
from compox.algorithm_utils.AlgorithmStorageMetrics import (
    AlgorithmStorageMetrics,
)
from compox.database_connection.InMemoryConnection import InMemoryConnection


@pytest.fixture
def valid_alg_dir(tmp_path):
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
    (directory / "pyproject.toml").write_text(toml.dumps(content))
    (directory / "Runner.py").write_text("class Runner: pass")
    return str(directory)


def test_algorithm_storage_metrics_deduplicate_shared_objects():
    db = InMemoryConnection()
    db.create_collections(
        [
            "algorithm-store",
            "module-store",
            "asset-store",
            "algorithm-checkpoint-store",
        ]
    )

    db.put_objects("module-store", ["module-1"], [b"module"])
    db.put_objects("asset-store", ["asset-1"], [b"asset"])
    checkpoint_manifest = {
        "checkpoint_id": "chk-1",
        "training_id": "train-1",
        "parent_algorithm_id": "alg-1",
        "created_at": "2026-04-01T12:00:00",
        "properties": {},
        "tags": [],
        "parent_checkpoint_id": None,
        "assets": {"weights.bin": "asset-1"},
    }
    db.put_objects(
        "algorithm-checkpoint-store",
        ["chk-1"],
        [json.dumps(checkpoint_manifest)],
    )

    algorithm_record = {
        "algorithm_id": "alg-1",
        "algorithm_name": "my_algo",
        "algorithm_major_version": "1",
        "latest_algorithm_minor_version": "1",
        "algorithm_minor_version": {
            "0": {
                "timestamp": "2026-04-01 12:00:00",
                "module_id": "module-1",
                "assets": {"weights.bin": "asset-1"},
            },
            "1": {
                "timestamp": "2026-04-01 12:00:01",
                "module_id": "module-1",
                "assets": {"weights.bin": "asset-1"},
            },
        },
        "checkpoints": ["chk-1"],
    }

    metrics = AlgorithmStorageMetrics(db).calculate_metrics(algorithm_record)

    expected_checkpoint_bytes = len(json.dumps(checkpoint_manifest).encode("utf-8"))
    assert metrics["module_count"] == 1
    assert metrics["asset_count"] == 1
    assert metrics["checkpoint_count"] == 1
    assert metrics["module_bytes"] == len(b"module")
    assert metrics["asset_bytes"] == len(b"asset")
    assert metrics["checkpoint_manifest_bytes"] == expected_checkpoint_bytes
    assert (
        metrics["logical_size_bytes"]
        == len(b"module") + len(b"asset") + expected_checkpoint_bytes
    )


def test_algorithm_deployer_persists_storage_metrics(valid_alg_dir):
    db = InMemoryConnection()
    deployer = AlgorithmDeployer(valid_alg_dir)

    algorithm_id = deployer.store_algorithm(database_connection=db)
    algorithm_key = (
        f"{algorithm_id}~{deployer.algorithm_name}~"
        f"{deployer.algorithm_major_version}"
    )
    stored_record = json.loads(
        db.get_objects("algorithm-store", [algorithm_key])[0]
    )

    assert "storage_metrics" in stored_record
    assert stored_record["storage_metrics"]["logical_size_bytes"] > 0
    assert stored_record["storage_metrics"]["module_count"] == 1


def test_algorithm_storage_metrics_ignore_missing_referenced_objects():
    db = InMemoryConnection()
    db.create_collections(
        [
            "algorithm-store",
            "module-store",
            "asset-store",
            "algorithm-checkpoint-store",
        ]
    )

    db.put_objects("module-store", ["module-1"], [b"module"])

    algorithm_record = {
        "algorithm_id": "alg-1",
        "algorithm_name": "my_algo",
        "algorithm_major_version": "1",
        "latest_algorithm_minor_version": "0",
        "algorithm_minor_version": {
            "0": {
                "timestamp": "2026-04-01 12:00:00",
                "module_id": "module-1",
                "assets": {"weights.bin": "missing-asset"},
            }
        },
        "checkpoints": ["missing-checkpoint"],
    }

    metrics = AlgorithmStorageMetrics(db).calculate_metrics(algorithm_record)

    assert metrics["module_count"] == 1
    assert metrics["asset_count"] == 0
    assert metrics["checkpoint_count"] == 0
    assert metrics["module_bytes"] == len(b"module")
    assert metrics["asset_bytes"] == 0
    assert metrics["checkpoint_manifest_bytes"] == 0
    assert metrics["logical_size_bytes"] == len(b"module")
