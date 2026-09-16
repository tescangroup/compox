"""
Copyright 2025 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import pytest
import numpy as np
import h5py
import io
from pydantic import ValidationError

from compox.training.TrainingSample import TrainingSample
from compox.database_connection.InMemoryConnection import InMemoryConnection
from compox.algorithm_utils.io_schemas import DataSchema
from compox.exceptions import CompoxNotFoundError


class MySchema(DataSchema):
    array: np.ndarray


def _post_file(r, name, db_connection) -> str:
    r = MySchema.model_validate(r)
    r = r.model_dump()
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        for key in r.keys():
            if r[key] is not None:
                f.create_dataset(
                    key,
                    data=r[key],
                )
    # upload response to minio
    db_connection.put_objects(
        "data-store",
        [name],
        [bio.getvalue()],
    )


@pytest.fixture
def db_connection():
    return InMemoryConnection()


def test_valid_training_sample(db_connection):
    """Test creating a valid training sample."""
    sample_manifest = TrainingSample(
        db_connection,
        sample_manifest={
            "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
            "files": [
                {"input": ["fid1", "fid1b"], "target": ["fid1_y"]},
                {"input": ["fid2"], "target": ["fid2_y"]},
                {"input": ["fid3"], "target": ["fid3_y"]},
            ],
            "tags": ["segmentation:skull", "author:jan"],
            "time_created": "2025-09-02T11:42:00+02:00",
        },
    )

    assert sample_manifest.sample_id == "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90"


@pytest.mark.parametrize(
    "sample_manifest",
    [
        (
            {
                "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
                "files": [
                    {"input": ["fid1", "fid1b"], "invalid_field": ["fid1_y"]},
                    {"input": ["fid2"], "target": ["fid2_y"]},
                    {"input": ["fid3"], "target": ["fid3_y"]},
                ],
                "tags": ["segmentation:skull", "author:jan"],
                "time_created": "2025-09-02T11:42:00+02:00",
            }
        ),
        (
            {
                "files": [
                    {"input": ["fid1", "fid1b"], "target": ["fid1_y"]},
                    {"input": ["fid2"], "target": ["fid2_y"]},
                    {"input": ["fid3"], "target": ["fid3_y"]},
                ],
                "tags": ["segmentation:skull", "author:jan"],
                "time_created": "2025-09-02T11:42:00+02:00",
            }
        ),
        (
            {
                "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
                "files": [
                    {"input": ["fid1", "fid1b"], "target": ["fid1_y"]},
                    {"input": ["fid2"], "target": ["fid2_y"]},
                    {"input": ["fid3"], "target": ["fid3_y"]},
                ],
                "tags": ["segmentation:skull", "author:jan"],
            }
        ),
        (
            {
                "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
                "tags": ["segmentation:skull", "author:jan"],
                "time_created": "2025-09-02T11:42:00+02:00",
            }
        ),
        (
            {
                "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
                "files": [],
                "tags": ["segmentation:skull", "author:jan"],
                "time_created": "2025-09-02T11:42:00+02:00",
            }
        ),
    ],
    ids=[
        "invalid field in files",
        "missing sample_id",
        "missing time_created",
        "missing files",
        "empty files list",
    ],
)
def test_invalid_training_sample(db_connection, sample_manifest):
    """Test creating an invalid training sample."""
    try:
        TrainingSample(db_connection, sample_manifest=sample_manifest)
        assert (
            False
        ), f"ValidationError was expected for {sample_manifest.get('sample_id', '')}"
    except ValidationError:
        pass


def test_len_returns_number_of_files(db_connection):
    """Test the length of the files list in the training sample."""
    sample_manifest = TrainingSample(
        db_connection,
        sample_manifest={
            "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
            "files": [
                {"input": ["fid1", "fid1b"], "target": ["fid1_y"]},
                {"input": ["fid2"], "target": ["fid2_y"]},
                {"input": ["fid3"], "target": ["fid3_y"]},
            ],
            "tags": ["segmentation:skull", "author:jan"],
            "time_created": "2025-09-02T11:42:00+02:00",
        },
    )

    assert len(sample_manifest) == 3


def test_get_file_list(db_connection):
    """Test retrieving the raw file list from the training sample."""
    sample_manifest = TrainingSample(
        db_connection,
        sample_manifest={
            "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
            "files": [
                {"input": ["fid1", "fid1b"], "target": ["fid1_y"]},
                {"input": ["fid2"], "target": ["fid2_y"]},
                {"input": ["fid3"], "target": ["fid3_y"]},
            ],
            "tags": ["segmentation:skull", "author:jan"],
            "time_created": "2025-09-02T11:42:00+02:00",
        },
    )

    file_list = sample_manifest.get_file_list()
    assert isinstance(file_list, list)
    assert len(file_list) == 3
    assert file_list[0] == {"input": ["fid1", "fid1b"], "target": ["fid1_y"]}
    assert file_list[1] == {"input": ["fid2"], "target": ["fid2_y"]}
    assert file_list[2] == {"input": ["fid3"], "target": ["fid3_y"]}


def test_save_load_delete_sample_manifest(db_connection):
    file_ids = [
        "fid1",
        "fid1b",
        "fid1_y",
        "fid2",
        "fid2_y",
        "fid3",
        "fid3_y",
        "fid4",
        "fid4_y",
        "fid5",
        "fid5b",
        "fid5_y",
        "fid6",
        "fid6_y",
        "fid7",
        "fid7_y",
        "fid8",
        "fid8_y",
        "fid9",
        "fid9_y",
    ]

    data_arrays = [np.random.rand(10, 10, 10) for _ in file_ids]
    for name, data in zip(file_ids, data_arrays):
        _post_file(
            {"array": data},
            name,
            db_connection,
        )
    sample_manifest = TrainingSample(
        db_connection,
        sample_manifest={
            "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
            "files": [
                {"input": ["fid1", "fid1b"], "target": ["fid1_y"]},
                {"input": ["fid2"], "target": ["fid2_y"]},
                {"input": ["fid3"], "target": ["fid3_y"]},
            ],
            "tags": ["segmentation:skull", "author:jan"],
            "time_created": "2025-09-02T11:42:00+02:00",
        },
    )
    assert (
        sample_manifest.save_sample_manifest()
    ), "Failed to save sample manifest."
    loaded_sample = TrainingSample(
        db_connection,
        sample_id="3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
    )
    assert (
        loaded_sample.load_sample_manifest()
    ), "Failed to load sample manifest."

    assert (
        loaded_sample.sample_manifest == sample_manifest.sample_manifest
    ), "Loaded sample manifest does not match the original."
    assert (
        loaded_sample.delete_sample_manifest()
    ), "Failed to delete sample manifest."
    with pytest.raises(CompoxNotFoundError) as exc_info:
        TrainingSample(
            db_connection,
            sample_id="3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
        ).load_sample_manifest()
    assert exc_info.value.code == "sample_not_found"


@pytest.mark.parametrize(
    "query_positive, query_negative, expected",
    [
        pytest.param(["stage:raw"], [], True, id="positive present"),
        pytest.param(
            ["owner:ml", "organ:skull"],
            [],
            True,
            id="multiple positives present",
        ),
        pytest.param(
            ["owner:ml", "missing:tag"], [], False, id="missing positive tag"
        ),
        pytest.param([], ["banned:tag"], True, id="negative not present -> ok"),
        pytest.param([], ["owner:ml"], False, id="negative present -> fail"),
        pytest.param(
            ["task:segmentation"],
            ["organ:skull"],
            False,
            id="positive ok but negative present",
        ),
        pytest.param([], [], True, id="no constraints -> always True"),
        pytest.param(
            ["stage:raw"],
            ["task:segmentation"],
            False,
            id="one positive one negative",
        ),
        pytest.param(
            ["task:segmentation"],
            ["nonexistent"],
            True,
            id="valid positive, safe negative",
        ),
    ],
)
def test_check_tags(db_connection, query_positive, query_negative, expected):
    """Test checking tags in the training sample."""
    sample_manifest = TrainingSample(
        db_connection,
        sample_manifest={
            "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
            "files": [{"input": ["f1"], "target": ["f1_y"]}],
            "tags": [
                "stage:raw",
                "owner:ml",
                "task:segmentation",
                "organ:skull",
            ],
            "time_created": "2025-09-02T11:42:00+02:00",
        },
    )
    assert (
        sample_manifest.check_tags(query_positive, query_negative) == expected
    )


def test_get_key_list(db_connection):
    """Test retrieving the list of unique keys from the training sample files."""
    sample_manifest = TrainingSample(
        db_connection,
        sample_manifest={
            "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
            "files": [
                {"input": ["fid1", "fid1b"], "target": ["fid1_y"]},
                {"input": ["fid2"], "target": ["fid2_y"]},
                {"input": ["fid3"], "target": ["fid3_y"]},
            ],
            "tags": ["segmentation:skull", "author:jan"],
            "time_created": "2025-09-02T11:42:00+02:00",
        },
    )

    key_list = sample_manifest.get_key_list()
    assert isinstance(key_list, list)
    assert len(key_list) == 2
    assert "input" in key_list
    assert "target" in key_list


def test_sample_indexing(db_connection):
    """Test indexing into the training sample to get individual file entries."""
    sample = TrainingSample(
        db_connection,
        sample_manifest={
            "sample_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
            "files": [
                {"input": ["fid1", "fid1b"], "target": ["fid1_y"]},
                {"input": ["fid2"], "target": ["fid2_y"]},
                {"input": ["fid3"], "target": ["fid3_y"]},
            ],
            "tags": ["segmentation:skull", "author:jan"],
            "time_created": "2025-09-02T11:42:00+02:00",
        },
    )

    assert sample[0] == {
        "input": ["fid1", "fid1b"],
        "target": ["fid1_y"],
    }, "Indexing by integer failed."
    assert sample[1:3] == [
        {"input": ["fid2"], "target": ["fid2_y"]},
        {"input": ["fid3"], "target": ["fid3_y"]},
    ], "Slicing by range failed."
    assert sample[:] == [
        {"input": ["fid1", "fid1b"], "target": ["fid1_y"]},
        {"input": ["fid2"], "target": ["fid2_y"]},
        {"input": ["fid3"], "target": ["fid3_y"]},
    ], "Full slice failed."
    assert sample[0, "input"] == [
        "fid1",
        "fid1b",
    ], "Indexing by integer and key failed."
    assert sample[:, "input"] == [
        "fid1",
        "fid1b",
        "fid2",
        "fid3",
    ], "Slicing by range and key failed."
    assert sample[:, :] == [
        "fid1",
        "fid1b",
        "fid1_y",
        "fid2",
        "fid2_y",
        "fid3",
        "fid3_y",
    ], "Slicing by range and key failed."
