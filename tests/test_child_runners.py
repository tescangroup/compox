"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from pydantic import ValidationError

from compox.algorithm_utils.Image2AlignmentRunner import (
    Image2AlignmentRunner,
)
from compox.algorithm_utils.io_schemas import MultiRegionSegmentationSchema
from compox.algorithm_utils.Image2EmbeddingRunner import (
    Image2EmbeddingRunner,
)
from compox.algorithm_utils.Image2ImageRunner import Image2ImageRunner
from compox.algorithm_utils.Image2MultiRegionSegmentationRunner import (
    Image2MultiRegionSegmentationRunner,
)
from compox.algorithm_utils.Image2SegmentationRunner import (
    Image2SegmentationRunner,
)
from compox.algorithm_utils.Segmentation2SegmentationRunner import (
    Segmentation2SegmentationRunner,
)
from compox.tasks.context_handler import current_handler


PREPROCESS_CONFIG = [
    # RunnerClass, key, attr_name, expected_val
    (Image2AlignmentRunner, "image", "_input_images_count", 3),
    (Image2EmbeddingRunner, "image", "_input_images_count", 3),
    (Image2ImageRunner, "image", "_input_images_count", 3),
    (
        Image2MultiRegionSegmentationRunner,
        "image",
        "_input_images_shape",
        (3, 2, 2),
    ),
    (Image2SegmentationRunner, "image", "_input_images_shape", (3, 2, 2)),
    (Segmentation2SegmentationRunner, "mask", "_input_images_count", 3),
]

POSTPROCESS_CONFIG = [
    # RunnerClass, key, attr_name, expected_val,
    # good_fn,
    # bad_fn,
    # exc
    (
        Image2AlignmentRunner,
        "transform_matrix",
        "_input_images_count",
        3,
        lambda cnt: [np.eye(3) * i for i in range(cnt - 1)],
        lambda cnt: np.eye(3),
        AssertionError,
    ),
    (
        Image2EmbeddingRunner,
        "features",
        "_input_images_count",
        3,
        lambda cnt: np.random.randn(cnt, 128),
        None,
        None,
    ),
    (
        Image2ImageRunner,
        "image",
        "_input_images_count",
        3,
        lambda cnt: np.zeros((cnt, 2, 2)),
        lambda cnt: np.zeros((cnt - 1, 2, 2)),
        ValueError,
    ),
    (
        Image2SegmentationRunner,
        "mask",
        "_input_images_shape",
        (3, 2, 2),
        lambda shape: np.zeros(shape),
        lambda shape: np.zeros((shape[0] - 1, *shape[1:])),
        ValueError,
    ),
    (
        Segmentation2SegmentationRunner,
        "mask",
        "_input_images_count",
        3,
        lambda cnt: np.zeros((cnt, 2, 2)),
        lambda cnt: np.zeros((cnt - 1, 2, 2)),
        ValueError,
    ),
]


@pytest.fixture
def fake_handler():
    """
    Fixture that provides a mocked TaskHandler.
        - fetch_data(*ids, keys…) returns a fresh list of 3 dicts, each mapping
        the last key to a 2×2 numpy array scaled by 10, 20, 30.
        - post_data(...) always returns ["id1", "id2"].

    Returns
    -------
    MagicMock
        A TaskHandler mock with fetch_data and post_data configured.
    """
    handler = MagicMock()

    def fake_fetch(ids, *keys, parallel=False, **kwargs):
        return [{keys[-1]: np.ones((2, 2)) * i} for i in [10, 20, 30]]

    handler.fetch_data.side_effect = fake_fetch

    handler.post_data.return_value = ["id1", "id2"]
    return handler


# Test 1 - Preprocessing of all Runners
@pytest.mark.parametrize(
    "RunnerClass, key, attr_name, expected_val", PREPROCESS_CONFIG
)
def test_preprocess_all_runners(
    RunnerClass, key, attr_name, expected_val, fake_handler
):
    """
    Verify that 'preprocess' correctly calls 'fetch_data' and returns numpy ndarray of correct shape
    """

    # define the RunnerClass inference method so that it is instantiable through abstact base class
    class RunnerClass(RunnerClass):
        def inference(self, data, args={}):
            return data

    current_handler.set(fake_handler)
    runner = RunnerClass.__new__(RunnerClass)
    runner.initialize(device="cpu")
    inp = {"input_dataset_ids": ["a", "b", "c"]}
    out = runner.preprocess(inp, args={})
    called_ids, *_, called_key = fake_handler.fetch_data.call_args[0]
    actual = getattr(runner, attr_name)

    assert called_ids == [
        "a",
        "b",
        "c",
    ], f"Expected '{RunnerClass.__name__}.input_dataset_ids' to be ['a', 'b', 'c'], got {called_ids!r}"
    assert (
        called_key == key
    ), f"Expected {RunnerClass.__name__!r} key be {key!r}, got {called_key!r}"
    assert isinstance(
        out, np.ndarray
    ), f"'{RunnerClass.__name__}.preprocess' should return 'np.ndarray', got {type(out)!r}"
    assert out.shape == (
        3,
        2,
        2,
    ), f"'{RunnerClass.__name__}.preprocess' should return result with shape '(3,2,2)', got {out.shape!r}"
    assert (
        actual == expected_val
    ), f"'{RunnerClass.__name__}.preprocess' should set atribute {attr_name!r} to {expected_val!r}, got {actual!r}"


# Test 2 - Postprocessing of all Runners
@pytest.mark.parametrize(
    "RunnerClass, key, attr_name, expected_val, good_fn, bad_fn, exc",
    POSTPROCESS_CONFIG,
)
def test_postprocess_all_runners(
    RunnerClass,
    key,
    attr_name,
    expected_val,
    good_fn,
    bad_fn,
    exc,
    fake_handler,
):
    """
    Verify that 'postprocess' correctly calls 'post_data' and returns dict
    """

    # define the RunnerClass inference method so that it is instantiable through abstact base class
    class RunnerClass(RunnerClass):
        def inference(self, data, args={}):
            return data

    current_handler.set(fake_handler)
    runner = RunnerClass.__new__(RunnerClass)
    runner.initialize(device="cpu")
    setattr(runner, attr_name, expected_val)
    good_input = good_fn(
        expected_val if isinstance(expected_val, int) else expected_val
    )
    out_ids = runner.postprocess(good_input, args={})
    posted = fake_handler.post_data.call_args[0][0]

    assert isinstance(
        posted, list
    ), f"{RunnerClass.__name__}.postprocess should call post_data with a list, got {type(posted)!r}"
    assert all(
        isinstance(d, dict) for d in posted
    ), f"{RunnerClass.__name__}.postprocess should pass a list of dicts, got {[type(d) for d in posted]!r}"
    assert out_ids == [
        "id1",
        "id2",
    ], f"Expected '{RunnerClass.__name__}.postprocess' to return '['id1', 'id2']', got {out_ids!r}"
    for idx, d in enumerate(posted):
        try:
            np.testing.assert_array_equal(
                d[key],
                (
                    good_input[idx]
                    if isinstance(good_input, list)
                    else good_input[idx]
                ),
            )
        except:
            (
                f"{RunnerClass.__name__}.postprocess: value under '{key}' at index {idx} does not match expected array"
            )

    if exc:
        bad_input = bad_fn(
            expected_val if isinstance(expected_val, int) else expected_val
        )
        with pytest.raises(exc):
            runner.postprocess(bad_input, args={})


def test_postprocess_image2_multi_region_segmentation_runner(fake_handler):
    """
    Verify that multi-region postprocess forwards list[dict] outputs unchanged.
    """

    class RunnerClass(Image2MultiRegionSegmentationRunner):
        def inference(self, data, args={}):
            return data

    current_handler.set(fake_handler)
    runner = RunnerClass.__new__(RunnerClass)
    runner.initialize(device="cpu")
    runner._input_images_shape = (3, 2, 2)

    good_input = [
        {
            "region_names": ["a", "b"],
            "region_masks": [np.zeros((2, 2)), np.ones((2, 2))],
        },
        {
            "region_names": ["c"],
            "region_masks": [np.full((2, 2), 2)],
        },
        {
            "region_names": [],
            "region_masks": [],
        },
    ]

    out_ids = runner.postprocess(good_input, args={})
    posted = fake_handler.post_data.call_args[0][0]

    assert posted == good_input
    assert out_ids == ["id1", "id2"]


def test_postprocess_image2_multi_region_segmentation_runner_converts_named_dicts(
    fake_handler,
):
    class RunnerClass(Image2MultiRegionSegmentationRunner):
        def inference(self, data, args={}):
            return data

    current_handler.set(fake_handler)
    runner = RunnerClass.__new__(RunnerClass)
    runner.initialize(device="cpu")
    runner._input_images_shape = (3, 2, 2)

    good_input = [
        {
            "mid_intensity": np.zeros((2, 2)),
            "high_intensity": np.ones((2, 2)),
        },
        {
            "foreground": np.full((2, 2), 2),
        },
        {},
    ]

    out_ids = runner.postprocess(good_input, args={})
    posted = fake_handler.post_data.call_args[0][0]

    assert posted[0]["region_names"] == ["mid_intensity", "high_intensity"]
    assert posted[1]["region_names"] == ["foreground"]
    assert posted[2]["region_names"] == []
    np.testing.assert_array_equal(
        posted[0]["region_masks"][0], good_input[0]["mid_intensity"]
    )
    np.testing.assert_array_equal(
        posted[0]["region_masks"][1], good_input[0]["high_intensity"]
    )
    np.testing.assert_array_equal(
        posted[1]["region_masks"][0], good_input[1]["foreground"]
    )
    assert posted[2]["region_masks"] == []
    assert out_ids == ["id1", "id2"]


def test_postprocess_image2_multi_region_segmentation_runner_rejects_bad_len(
    fake_handler,
):
    class RunnerClass(Image2MultiRegionSegmentationRunner):
        def inference(self, data, args={}):
            return data

    current_handler.set(fake_handler)
    runner = RunnerClass.__new__(RunnerClass)
    runner.initialize(device="cpu")
    runner._input_images_shape = (3, 2, 2)

    bad_input = [
        {"region_names": ["a"], "region_masks": [np.zeros((2, 2))]},
        {"region_names": ["b"], "region_masks": [np.ones((2, 2))]},
    ]

    with pytest.raises(ValueError):
        runner.postprocess(bad_input, args={})


def test_postprocess_image2_multi_region_segmentation_runner_rejects_partial_explicit_item(
    fake_handler,
):
    class RunnerClass(Image2MultiRegionSegmentationRunner):
        def inference(self, data, args={}):
            return data

    current_handler.set(fake_handler)
    runner = RunnerClass.__new__(RunnerClass)
    runner.initialize(device="cpu")
    runner._input_images_shape = (3, 2, 2)

    bad_input = [
        {"region_names": ["a"], "region_masks": [np.zeros((2, 2))]},
        {"region_names": ["b"]},
        {"region_names": ["c"], "region_masks": [np.ones((2, 2))]},
    ]

    with pytest.raises(ValueError):
        runner.postprocess(bad_input, args={})


def test_postprocess_image2_multi_region_segmentation_runner_rejects_bad_item(
    fake_handler,
):
    class RunnerClass(Image2MultiRegionSegmentationRunner):
        def inference(self, data, args={}):
            return data

    current_handler.set(fake_handler)
    runner = RunnerClass.__new__(RunnerClass)
    runner.initialize(device="cpu")
    runner._input_images_shape = (3, 2, 2)

    bad_input = [
        {"region_names": ["a"], "region_masks": [np.zeros((2, 2))]},
        {"region_names": ["b"], "region_masks": []},
        {"region_names": ["c"], "region_masks": [np.ones((2, 2))]},
    ]

    with pytest.raises(ValueError):
        runner.postprocess(bad_input, args={})


def test_multi_region_segmentation_schema_rejects_non_array_mask():
    with pytest.raises(ValidationError):
        MultiRegionSegmentationSchema.model_validate(
            {
                "region_names": ["a"],
                "region_masks": [[[0, 1], [1, 0]]],
            }
        )


def test_multi_region_segmentation_schema_rejects_non_2d_mask():
    with pytest.raises(ValidationError):
        MultiRegionSegmentationSchema.model_validate(
            {
                "region_names": ["a"],
                "region_masks": [np.zeros((2, 2, 2))],
            }
        )


def test_multi_region_segmentation_schema_rejects_mismatched_mask_shapes():
    with pytest.raises(ValidationError):
        MultiRegionSegmentationSchema.model_validate(
            {
                "region_names": ["a", "b"],
                "region_masks": [
                    np.zeros((2, 2)),
                    np.zeros((3, 3)),
                ],
            }
        )
