"""
Copyright 2026 Tescan group, a.s.
All rights reserved
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from compox.components.builtin_algorithm_importer import (
    BuiltinAlgorithmImporter,
)
from compox.components.compox_algorithm_bundle_builder import (
    CompoxAlgorithmBundleBuilder,
)
from compox.database_connection.CompoxAlgorithmBundleConnection import (
    CompoxAlgorithmBundleConnection,
)
from compox.database_connection.InMemoryConnection import InMemoryConnection
from compox.exceptions import CompoxBundleError


def _make_settings(bundle_path: Path, bundle_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        storage=SimpleNamespace(
            builtin_storage_bundle_path=str(bundle_path),
            builtin_storage_bundle_key=bundle_key,
        )
    )


def _seed_source_db(
    include_checkpoint: bool = True,
    algorithm_id: str = "alg-1",
    module_id: str = "module-1",
    asset_id: str = "asset-1",
    module_bytes: bytes = b"module-bytes",
    asset_bytes: bytes = b"asset-bytes",
) -> InMemoryConnection:
    db = InMemoryConnection()
    db.create_collections(
        [
            "algorithm-store",
            "module-store",
            "asset-store",
            "algorithm-checkpoint-store",
        ]
    )

    algorithm_record = {
        "algorithm_id": algorithm_id,
        "algorithm_name": "my_algo",
        "algorithm_major_version": "1",
        "latest_algorithm_minor_version": "0",
        "algorithm_minor_version": {
            "0": {
                "timestamp": "2026-03-30 12:00:00",
                "module_id": module_id,
                "assets": {"weights.bin": asset_id},
            }
        },
        "checkpoints": ["chk-1"] if include_checkpoint else [],
    }
    db.put_objects(
        "algorithm-store",
        [f"{algorithm_id}~my_algo~1"],
        [json.dumps(algorithm_record)],
    )
    db.put_objects("module-store", [module_id], [module_bytes])
    db.put_objects("asset-store", [asset_id], [asset_bytes])

    if include_checkpoint:
        checkpoint_manifest = {
            "checkpoint_id": "chk-1",
            "parent_algorithm_id": algorithm_id,
            "training_id": "train-1",
            "assets": {"weights.bin": asset_id},
            "created_at": "2026-03-30T12:00:00Z",
            "properties": {},
            "tags": [],
            "parent_checkpoint_id": None,
        }
        db.put_objects(
            "algorithm-checkpoint-store",
            ["chk-1"],
            [json.dumps(checkpoint_manifest)],
        )

    return db


def test_build_bundle_exposes_metadata_and_counts(tmp_path):
    """Bundle metadata and object counts should be available via the reader."""
    source_db = _seed_source_db(include_checkpoint=True)
    bundle_key = CompoxAlgorithmBundleBuilder.generate_key()
    bundle_path = tmp_path / "bundle.zip"

    builder = CompoxAlgorithmBundleBuilder(source_db, bundle_key)
    object_count = builder.build(str(bundle_path))
    bundle = CompoxAlgorithmBundleConnection(str(bundle_path), bundle_key)
    info = bundle.get_bundle_info()

    assert object_count == 4
    assert info["format"] == "compox-migration-bundle-v1"
    assert info["metadata"]["bundle_kind"] == "builtin_algorithm_snapshot"
    assert "created_at" in info["metadata"]
    assert "compox_version" in info["metadata"]
    assert "content_sha256" in info["metadata"]
    assert info["object_count"] == 4
    assert info["object_counts_by_collection"]["algorithm-store"] == 1
    assert info["object_counts_by_collection"]["module-store"] == 1
    assert info["object_counts_by_collection"]["asset-store"] == 1
    assert (
        info["object_counts_by_collection"]["algorithm-checkpoint-store"] == 1
    )


def test_build_bundle_declares_empty_checkpoint_collection(tmp_path):
    """Empty checkpoint store should still be declared in bundle collections."""
    source_db = _seed_source_db(include_checkpoint=False)
    bundle_key = CompoxAlgorithmBundleBuilder.generate_key()
    bundle_path = tmp_path / "bundle_empty_checkpoint.zip"

    builder = CompoxAlgorithmBundleBuilder(source_db, bundle_key)
    builder.build(str(bundle_path))
    bundle = CompoxAlgorithmBundleConnection(str(bundle_path), bundle_key)

    assert "algorithm-checkpoint-store" in bundle.list_collections()
    assert bundle.list_objects("algorithm-checkpoint-store") == []


def test_builtin_algorithm_importer_imports_bundle_snapshot(tmp_path):
    """Importer should copy the bundled snapshot into the target backend."""
    source_db = _seed_source_db(include_checkpoint=True)
    target_db = InMemoryConnection()
    target_db.create_collections(
        [
            "system-store",
            "algorithm-store",
            "module-store",
            "asset-store",
            "algorithm-checkpoint-store",
        ]
    )

    bundle_key = CompoxAlgorithmBundleBuilder.generate_key()
    bundle_path = tmp_path / "import_bundle.zip"
    builder = CompoxAlgorithmBundleBuilder(source_db, bundle_key)
    builder.build(str(bundle_path))

    importer = BuiltinAlgorithmImporter(
        target_db, _make_settings(bundle_path, bundle_key)
    )
    importer.run_startup_migration()

    imported_algorithm = json.loads(
        target_db.get_objects("algorithm-store", ["alg-1~my_algo~1"])[0]
    )
    import_state = json.loads(
        target_db.get_objects("system-store", ["migration-state"])[0]
    )

    assert (
        target_db.get_objects("module-store", ["module-1"])[0]
        == b"module-bytes"
    )
    assert (
        target_db.get_objects("asset-store", ["asset-1"])[0] == b"asset-bytes"
    )
    assert (
        json.loads(
            target_db.get_objects("algorithm-checkpoint-store", ["chk-1"])[0]
        )["checkpoint_id"]
        == "chk-1"
    )
    checkpoint_manifest = json.loads(
        target_db.get_objects("algorithm-checkpoint-store", ["chk-1"])[0]
    )
    assert "source_backend_version" in imported_algorithm
    assert checkpoint_manifest["training_id"] == "train-1"
    assert checkpoint_manifest["properties"] == {}
    assert import_state["last_import_status"] == "COMPLETED"
    assert "last_imported_bundle_sha256" in import_state


def test_builtin_algorithm_importer_reuses_existing_builtin_id_and_minor(
    tmp_path,
):
    """
    Existing builtin algorithm should keep its id and not create a new minor
    when the imported latest module/assets match the current latest version.
    """
    source_db = _seed_source_db(
        include_checkpoint=True,
        algorithm_id="source-alg",
        module_id="module-1",
        asset_id="asset-1",
    )
    target_db = InMemoryConnection()
    target_db.create_collections(
        [
            "system-store",
            "algorithm-store",
            "module-store",
            "asset-store",
            "algorithm-checkpoint-store",
        ]
    )
    target_record = {
        "algorithm_id": "target-alg",
        "algorithm_name": "my_algo",
        "algorithm_major_version": "1",
        "latest_algorithm_minor_version": "0",
        "algorithm_minor_version": {
            "0": {
                "timestamp": "2026-03-29 12:00:00",
                "module_id": "module-1",
                "assets": {"weights.bin": "asset-1"},
            }
        },
        "checkpoints": [],
    }
    target_db.put_objects(
        "algorithm-store",
        ["target-alg~my_algo~1"],
        [json.dumps(target_record)],
    )
    target_db.put_objects("module-store", ["module-1"], [b"module-bytes"])
    target_db.put_objects("asset-store", ["asset-1"], [b"asset-bytes"])

    bundle_key = CompoxAlgorithmBundleBuilder.generate_key()
    bundle_path = tmp_path / "merge_same_latest.zip"
    CompoxAlgorithmBundleBuilder(source_db, bundle_key).build(str(bundle_path))

    importer = BuiltinAlgorithmImporter(
        target_db, _make_settings(bundle_path, bundle_key)
    )
    importer.run_startup_migration()

    merged_record = json.loads(
        target_db.get_objects("algorithm-store", ["target-alg~my_algo~1"])[0]
    )
    checkpoint_manifest = json.loads(
        target_db.get_objects("algorithm-checkpoint-store", ["chk-1"])[0]
    )

    assert merged_record["algorithm_id"] == "target-alg"
    assert merged_record["latest_algorithm_minor_version"] == "0"
    assert list(merged_record["algorithm_minor_version"].keys()) == ["0"]
    assert merged_record["checkpoints"] == ["chk-1"]
    assert checkpoint_manifest["parent_algorithm_id"] == "target-alg"


def test_builtin_algorithm_importer_appends_minor_for_new_latest_payload(
    tmp_path,
):
    """
    Existing builtin algorithm should append a new minor when the imported
    latest module/assets differ from the current latest version.
    """
    source_db = _seed_source_db(
        include_checkpoint=False,
        algorithm_id="source-alg",
        module_id="module-2",
        asset_id="asset-2",
        module_bytes=b"module-two",
        asset_bytes=b"asset-two",
    )
    target_db = InMemoryConnection()
    target_db.create_collections(
        [
            "system-store",
            "algorithm-store",
            "module-store",
            "asset-store",
            "algorithm-checkpoint-store",
        ]
    )
    target_record = {
        "algorithm_id": "target-alg",
        "algorithm_name": "my_algo",
        "algorithm_major_version": "1",
        "latest_algorithm_minor_version": "0",
        "algorithm_minor_version": {
            "0": {
                "timestamp": "2026-03-29 12:00:00",
                "module_id": "module-1",
                "assets": {"weights.bin": "asset-1"},
            }
        },
        "checkpoints": [],
    }
    target_db.put_objects(
        "algorithm-store",
        ["target-alg~my_algo~1"],
        [json.dumps(target_record)],
    )
    target_db.put_objects("module-store", ["module-1"], [b"module-one"])
    target_db.put_objects("asset-store", ["asset-1"], [b"asset-one"])

    bundle_key = CompoxAlgorithmBundleBuilder.generate_key()
    bundle_path = tmp_path / "merge_new_latest.zip"
    CompoxAlgorithmBundleBuilder(source_db, bundle_key).build(str(bundle_path))

    importer = BuiltinAlgorithmImporter(
        target_db, _make_settings(bundle_path, bundle_key)
    )
    importer.run_startup_migration()

    merged_record = json.loads(
        target_db.get_objects("algorithm-store", ["target-alg~my_algo~1"])[0]
    )

    assert merged_record["algorithm_id"] == "target-alg"
    assert merged_record["latest_algorithm_minor_version"] == "1"
    assert merged_record["algorithm_minor_version"]["1"]["module_id"] == "module-2"
    assert (
        merged_record["algorithm_minor_version"]["1"]["assets"]["weights.bin"]
        == "asset-2"
    )
    assert target_db.get_objects("module-store", ["module-2"])[0] == b"module-two"
    assert target_db.get_objects("asset-store", ["asset-2"])[0] == b"asset-two"


def test_builtin_algorithm_importer_merges_existing_algorithm_without_origin(
    tmp_path,
):
    """
    Existing algorithms are merged by normal versioning semantics even without
    any ownership metadata.
    """
    source_db = _seed_source_db(
        include_checkpoint=True,
        algorithm_id="source-alg",
        module_id="module-src",
        asset_id="asset-src",
        module_bytes=b"module-src-bytes",
        asset_bytes=b"asset-src-bytes",
    )
    target_db = InMemoryConnection()
    target_db.create_collections(
        [
            "system-store",
            "algorithm-store",
            "module-store",
            "asset-store",
            "algorithm-checkpoint-store",
        ]
    )
    existing_record = {
        "algorithm_id": "existing-alg",
        "algorithm_name": "my_algo",
        "algorithm_major_version": "1",
        "latest_algorithm_minor_version": "0",
        "algorithm_minor_version": {
            "0": {
                "timestamp": "2026-03-29 12:00:00",
                "module_id": "module-user",
                "assets": {"weights.bin": "asset-user"},
            }
        },
        "checkpoints": [],
    }
    target_db.put_objects(
        "algorithm-store",
        ["existing-alg~my_algo~1"],
        [json.dumps(existing_record)],
    )

    bundle_key = CompoxAlgorithmBundleBuilder.generate_key()
    bundle_path = tmp_path / "merge_existing_without_origin.zip"
    CompoxAlgorithmBundleBuilder(source_db, bundle_key).build(str(bundle_path))

    importer = BuiltinAlgorithmImporter(
        target_db, _make_settings(bundle_path, bundle_key)
    )
    importer.run_startup_migration()

    merged_record = json.loads(
        target_db.get_objects("algorithm-store", ["existing-alg~my_algo~1"])[0]
    )

    assert merged_record["algorithm_id"] == "existing-alg"
    assert merged_record["latest_algorithm_minor_version"] == "1"
    assert merged_record["algorithm_minor_version"]["1"]["module_id"] == "module-src"
    assert target_db.check_objects_exist("module-store", ["module-src"])[0] is True
    assert target_db.check_objects_exist("asset-store", ["asset-src"])[0] is True
    assert (
        json.loads(
            target_db.get_objects("algorithm-checkpoint-store", ["chk-1"])[0]
        )["parent_algorithm_id"]
        == "existing-alg"
    )


def test_builtin_algorithm_importer_fails_hard_on_invalid_bundle_algorithm(
    tmp_path,
):
    """
    Invalid bundled algorithms should fail the import run and write FAILED state.
    """
    source_db = _seed_source_db(include_checkpoint=False)
    broken_record = json.loads(
        source_db.get_objects("algorithm-store", ["alg-1~my_algo~1"])[0]
    )
    broken_record["checkpoints"] = ["missing-checkpoint"]
    source_db.put_objects(
        "algorithm-store",
        ["alg-1~my_algo~1"],
        [json.dumps(broken_record)],
    )

    target_db = InMemoryConnection()
    target_db.create_collections(
        [
            "system-store",
            "algorithm-store",
            "module-store",
            "asset-store",
            "algorithm-checkpoint-store",
        ]
    )

    bundle_key = CompoxAlgorithmBundleBuilder.generate_key()
    bundle_path = tmp_path / "broken_bundle.zip"
    CompoxAlgorithmBundleBuilder(source_db, bundle_key).build(str(bundle_path))

    importer = BuiltinAlgorithmImporter(
        target_db, _make_settings(bundle_path, bundle_key)
    )

    try:
        importer.run_startup_migration()
        assert False, "Expected invalid bundled algorithm import to raise."
    except CompoxBundleError as e:
        assert e.code == "builtin_algorithm_import_failed"

    import_state = json.loads(
        target_db.get_objects("system-store", ["migration-state"])[0]
    )
    assert import_state["last_import_status"] == "FAILED"
    assert import_state["failed_algorithms"] == 1
    assert import_state["error_code"] == "builtin_algorithm_import_failed"
    assert import_state["rollback_status"] == "COMPLETED"
    assert (
        target_db.check_objects_exist("algorithm-store", ["alg-1~my_algo~1"])[0]
        is False
    )
    assert (
        target_db.check_objects_exist("module-store", ["module-1"])[0] is False
    )
    assert (
        target_db.check_objects_exist("asset-store", ["asset-1"])[0] is False
    )
    assert (
        target_db.check_objects_exist(
            "algorithm-checkpoint-store", ["missing-checkpoint"]
        )[0]
        is False
    )


def test_algorithm_bundle_end_to_end_import(tmp_path):
    """
    Bundle build and import should work end-to-end using in-memory source and
    target backends, including checkpoint parent algorithm rewrite.
    """
    source_db = _seed_source_db()
    target_db = InMemoryConnection()
    target_db.create_collections(
        [
            "system-store",
            "algorithm-store",
            "module-store",
            "asset-store",
            "algorithm-checkpoint-store",
        ]
    )

    bundle_key = CompoxAlgorithmBundleBuilder.generate_key()
    bundle_path = tmp_path / "filesystem_bundle.zip"
    CompoxAlgorithmBundleBuilder(source_db, bundle_key).build(str(bundle_path))

    importer = BuiltinAlgorithmImporter(
        target_db, _make_settings(bundle_path, bundle_key)
    )
    importer.run_startup_migration()

    imported_algorithm = json.loads(
        target_db.get_objects("algorithm-store", ["alg-1~my_algo~1"])[0]
    )
    imported_checkpoint = json.loads(
        target_db.get_objects("algorithm-checkpoint-store", ["chk-1"])[0]
    )
    import_state = json.loads(
        target_db.get_objects("system-store", ["migration-state"])[0]
    )

    assert imported_algorithm["algorithm_id"] == "alg-1"
    assert imported_checkpoint["parent_algorithm_id"] == "alg-1"
    assert target_db.get_objects("module-store", ["module-1"])[0] == b"module-bytes"
    assert target_db.get_objects("asset-store", ["asset-1"])[0] == b"asset-bytes"
    assert import_state["last_import_status"] == "COMPLETED"


def test_algorithm_bundle_end_to_end_update_appends_minor(tmp_path):
    """
    Importing an updated bundle into an existing builtin algorithm should append
    a new minor version using standard deploy semantics.
    """
    source_db = _seed_source_db(
        include_checkpoint=False,
        algorithm_id="source-alg",
        module_id="module-2",
        asset_id="asset-2",
        module_bytes=b"module-two",
        asset_bytes=b"asset-two",
    )
    target_db = InMemoryConnection()
    target_db.create_collections(
        [
            "system-store",
            "algorithm-store",
            "module-store",
            "asset-store",
            "algorithm-checkpoint-store",
        ]
    )
    target_record = {
        "algorithm_id": "target-alg",
        "algorithm_name": "my_algo",
        "algorithm_major_version": "1",
        "latest_algorithm_minor_version": "0",
        "algorithm_minor_version": {
            "0": {
                "timestamp": "2026-03-29 12:00:00",
                "module_id": "module-1",
                "assets": {"weights.bin": "asset-1"},
            }
        },
        "checkpoints": [],
    }
    target_db.put_objects(
        "algorithm-store",
        ["target-alg~my_algo~1"],
        [json.dumps(target_record)],
    )
    target_db.put_objects("module-store", ["module-1"], [b"module-one"])
    target_db.put_objects("asset-store", ["asset-1"], [b"asset-one"])

    bundle_key = CompoxAlgorithmBundleBuilder.generate_key()
    bundle_path = tmp_path / "filesystem_update_bundle.zip"
    CompoxAlgorithmBundleBuilder(source_db, bundle_key).build(str(bundle_path))

    importer = BuiltinAlgorithmImporter(
        target_db, _make_settings(bundle_path, bundle_key)
    )
    importer.run_startup_migration()

    merged_record = json.loads(
        target_db.get_objects("algorithm-store", ["target-alg~my_algo~1"])[0]
    )

    assert merged_record["latest_algorithm_minor_version"] == "1"
    assert merged_record["algorithm_minor_version"]["1"]["module_id"] == "module-2"
    assert target_db.get_objects("module-store", ["module-2"])[0] == b"module-two"
    assert target_db.get_objects("asset-store", ["asset-2"])[0] == b"asset-two"


def test_algorithm_bundle_end_to_end_rollback_on_failure(tmp_path):
    """
    A broken bundle import should fail and roll back in-memory target writes.
    """
    source_db = _seed_source_db(include_checkpoint=False)
    broken_record = json.loads(
        source_db.get_objects("algorithm-store", ["alg-1~my_algo~1"])[0]
    )
    broken_record["checkpoints"] = ["missing-checkpoint"]
    source_db.put_objects(
        "algorithm-store",
        ["alg-1~my_algo~1"],
        [json.dumps(broken_record)],
    )

    target_db = InMemoryConnection()
    target_db.create_collections(
        [
            "system-store",
            "algorithm-store",
            "module-store",
            "asset-store",
            "algorithm-checkpoint-store",
        ]
    )

    bundle_key = CompoxAlgorithmBundleBuilder.generate_key()
    bundle_path = tmp_path / "filesystem_broken_bundle.zip"
    CompoxAlgorithmBundleBuilder(source_db, bundle_key).build(str(bundle_path))

    importer = BuiltinAlgorithmImporter(
        target_db, _make_settings(bundle_path, bundle_key)
    )

    try:
        importer.run_startup_migration()
        assert False, "Expected broken filesystem bundle import to raise."
    except CompoxBundleError as e:
        assert e.code == "builtin_algorithm_import_failed"

    import_state = json.loads(
        target_db.get_objects("system-store", ["migration-state"])[0]
    )
    assert import_state["last_import_status"] == "FAILED"
    assert import_state["error_code"] == "builtin_algorithm_import_failed"
    assert import_state["rollback_status"] == "COMPLETED"
    assert target_db.check_objects_exist("algorithm-store", ["alg-1~my_algo~1"])[0] is False
    assert target_db.check_objects_exist("module-store", ["module-1"])[0] is False
    assert target_db.check_objects_exist("asset-store", ["asset-1"])[0] is False
