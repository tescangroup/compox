"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

import os
import numpy as np
import time

from compox.algorithm_debug import debug
from compox.algorithm_utils.BaseRunner import BaseRunner
from compox.algorithm_utils.io_schemas import GenericSchema
from dependencies.utils import scale_and_shift


class Runner(BaseRunner):
    """
    Minimal generic runner template.
    """

    def load_assets(self):
        """
        Load packaged assets once and cache them on the runner instance.
        """
        return None

    def preprocess(
        self, input_data: dict, args: dict | None = None
    ) -> list[dict]:
        """
        Fetch generic input datasets and pass them to inference.
        """
        self.log_message("Preprocessing generic input data.")
        datasets = self.fetch_data(
            input_data["input_dataset_ids"], GenericSchema
        )
        return datasets

    def inference(
        self, data: list[dict], args: dict | None = None
    ) -> list[dict]:
        """
        Apply a simple transform to each input dataset.
        """
        args = {} if args is None else args
        scale = float(args.get("scale", 1.0))
        bias = float(args.get("bias", 0.0))

        self.log_message(
            f"Running generic inference with scale={scale} and bias={bias}."
        )

        outputs = []
        total = max(len(data), 1)
        for i, item in enumerate(data):
            outputs.append({"data": scale_and_shift(item["data"], scale, bias)})
            self.set_progress((i + 1) / total)

        return outputs

    def postprocess(
        self, inference_output: list[dict], args: dict | None = None
    ) -> list[str]:
        """
        Upload the transformed outputs back to Compox storage.
        """
        self.log_message("Postprocessing generic inference output.")
        return self.post_data(inference_output, GenericSchema)

    def benchmark(self, args: dict | None = None) -> dict:
        """
        Benchmark algorithm performance on synthetic input.
        """
        dummy_data = np.random.rand(100, 256, 256)  # Example dummy data

        self.log_message(
            f"Benchmarking generic algorithm with dummy data shape: {dummy_data.shape}"
        )

        start_time = time.time()
        self.inference([{"data": dummy_data}], args)
        inference_time = time.time() - start_time

        return {"inference_time": inference_time}


if __name__ == "__main__":
    debug(
        algo_dir=os.path.dirname(__file__),
        data="path to data",
        params={"scale": 2.0, "bias": 1.0},
        device="cpu",
    )
