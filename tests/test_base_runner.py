"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import pytest
from unittest.mock import MagicMock

from compox.algorithm_utils.BaseRunner import BaseRunner
from compox.exceptions import CompoxStateError


def fake_preprocess(input_data, args):
    """
    Simulate preprocessing by annotating input_data with a 'pre' key.
    """
    input_data["pre"] = f"pre {args}"
    return input_data


def fake_inference(input_data, args):
    """
    Simulate inference by annotating input_data with an 'inf' key.
    """
    input_data["inf"] = f"inf {args}"
    return input_data


def fake_postprocess(input_data, args):
    """
    Simulate postprocessing by annotating input_data with a 'post' key.
    """
    input_data["post"] = f"post {args}"
    return input_data


@pytest.fixture
def base_runner(task_handler):
    """
    Provide a BaseRunner instance using the CPU device and the task_handler fixture.
    """

    class Runner(BaseRunner):
        """
        A mock runner class for testing purposes.
        """

        def __init__(self):
            pass

        def preprocess(self):
            pass

        def inference(self):
            pass

        def postprocess(self):
            pass

    task_handler.set_as_current_handler()
    base_runner = Runner.__new__(Runner)
    base_runner.initialize(device="cpu")
    base_runner._load_assets()
    return base_runner


# Test 1 - Run (preprocess, inference, postprocess)
def test_run(base_runner, monkeypatch, task_handler):
    """
    Verify that 'run()' correctly calls:
        - preprocess
        - inference
        - posprocess
    """
    input_data = {"data": "input_data"}
    args = {"arg": 2}

    monkeypatch.setattr(base_runner, "preprocess", fake_preprocess)
    monkeypatch.setattr(base_runner, "inference", fake_inference)
    monkeypatch.setattr(base_runner, "postprocess", fake_postprocess)

    task_handler.mark_as_completed = MagicMock()
    result = base_runner.run(input_data, args)

    assert result == None, f"Expected 'run()' to return 'None', got {result!r}"
    try:
        task_handler.mark_as_completed.assert_called_once_with(
            {
                "data": "input_data",
                "pre": "pre {'arg': 2}",
                "inf": "inf {'arg': 2}",
                "post": "post {'arg': 2}",
            }
        )
    except AssertionError as e:
        pytest.fail(f"mark_as_completed not called correctly: {e}")


# Test 2 - Test preprocess, inference and postprocess
def test_preprocess_inference_postprocess(base_runner, monkeypatch):
    """
    Verify that 'preprocess', 'inference' and 'postprocess' returns correct data
    """
    input_data = {"data": "input_data"}
    args = {"arg": 2}

    monkeypatch.setattr(base_runner, "preprocess", fake_preprocess)
    monkeypatch.setattr(base_runner, "inference", fake_inference)
    monkeypatch.setattr(base_runner, "postprocess", fake_postprocess)

    preprocess_output = base_runner.preprocess_base(input_data, args)
    inference_output = base_runner.inference_base(preprocess_output, args)
    postprocess_output = base_runner.postprocess_base(inference_output, args)

    assert (
        preprocess_output["pre"] == f"pre {args}"
    ), f"Expected 'preprocess_base' to return 'pre {args}', got {preprocess_output['pre']!r}"
    assert (
        inference_output["inf"] == f"inf {args}"
    ), f"Expected 'inference_base' to return 'inf {args}', got {inference_output['inf']!r}"
    assert (
        postprocess_output["post"] == f"post {args}"
    ), f"Expected 'postprocess_base' to return 'post {args}', got {postprocess_output['post']!r}"
    assert (
        len(postprocess_output) == 4
    ), f"Expected 'postprocess_base' to return 4 keys ['data', 'pre', 'inf', 'post'], got {list(postprocess_output.keys())}"


def test_training_helpers_require_training_handler(base_runner):
    """
    Verify training-only runner helpers fail with a typed state error when used
    with a non-training task handler.
    """
    with pytest.raises(CompoxStateError) as exc_info:
        base_runner.save_checkpoint({"weights": b"123"})

    assert exc_info.value.code == "training_handler_required"


# Test 3 - run_benchmark calls benchmark() and persists results
def test_run_benchmark_calls_benchmark_and_completes(base_runner, monkeypatch, task_handler):
    """
    Verify that 'run_benchmark()' calls benchmark() and forwards the returned
    dict to task_handler.mark_as_completed.
    """
    expected_results = {"latency_s": 0.5, "throughput": 2.0}

    def fake_benchmark(args):
        return expected_results

    monkeypatch.setattr(base_runner, "benchmark", fake_benchmark)
    task_handler.mark_as_completed = MagicMock()

    result = base_runner.run_benchmark(args={"iterations": 1})

    assert result is None, f"Expected 'run_benchmark()' to return None, got {result!r}"
    task_handler.mark_as_completed.assert_called_once_with(expected_results)
