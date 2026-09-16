"""
Copyright 2026 TESCAN GROUP, a.s.
All rights reserved
"""

import numpy as np

from compox.algorithm_utils.BaseRunner import BaseRunner
from compox.algorithm_utils.io_schemas import (
    ImageSchema,
    MultiRegionSegmentationSchema,
)


class Image2MultiRegionSegmentationRunner(BaseRunner):
    """
    A child class of BaseRunner that is used to run image to multi-region
    segmentation tasks.
    """

    algorithm_type = "Image2MultiRegionSegmentation"

    def fetch_data(
        self,
        file_ids: list[str],
        *keys: str,
        parallel: bool = False,
    ) -> list[dict]:
        """
        Fetches the data from the database and validates it using ImageSchema.
        """
        return self.task_handler.fetch_data(
            file_ids, ImageSchema, *keys, parallel=parallel
        )

    def post_data(self, data: list[dict], parallel: bool = False) -> list[str]:
        """
        Uploads a list of multi-region segmentation datasets to the database and
        validates them using MultiRegionSegmentationSchema.
        """
        return self.task_handler.post_data(
            data, MultiRegionSegmentationSchema, parallel
        )

    def preprocess(
        self, input_data: dict, args: dict | None = None
    ) -> np.ndarray:
        """
        Default Image2MultiRegionSegmentation preprocessing method.
        """
        input_images = self.fetch_data(input_data["input_dataset_ids"], "image")

        for i in range(len(input_images)):
            input_images[i] = input_images[i]["image"]

        input_images = np.stack(input_images, axis=0)

        self._input_images_shape = input_images.shape

        return input_images

    def postprocess(
        self, data: list[dict], args: dict | None = None
    ) -> list[str]:
        """
        Default Image2MultiRegionSegmentation postprocessing method.

        The inference method is expected to return one dictionary per input
        image. Each dictionary may be provided in one of two forms:
            - {"region_names": [...], "region_masks": [...]}
            - {"region_a": mask_a, "region_b": mask_b, ...}
        """
        if not isinstance(data, list):
            raise TypeError(
                "Data is not a list, please make sure that the inference "
                "method returns a list of dictionaries."
            )

        if len(data) != self._input_images_shape[0]:
            raise ValueError(
                f"Number of outputs {len(data)} does not correspond to the "
                f"input images shape {self._input_images_shape}."
            )

        normalized_data = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise TypeError(
                    "The data returned from the inference should be a list of "
                    f"dictionaries, but the element on position {i} is a "
                    f"{type(item)}."
                )
            normalized_data.append(self._normalize_output_item(item))

        return self.post_data(normalized_data)

    def _normalize_output_item(self, item: dict) -> dict:
        if "region_names" in item or "region_masks" in item:
            if "region_names" not in item or "region_masks" not in item:
                raise ValueError(
                    "Each output item using the explicit multi-region format "
                    "must contain both 'region_names' and 'region_masks' keys."
                )
            normalized_item = {
                "region_names": list(item["region_names"]),
                "region_masks": list(item["region_masks"]),
            }
        else:
            normalized_item = {
                "region_names": list(item.keys()),
                "region_masks": list(item.values()),
            }

        if len(normalized_item["region_names"]) != len(
            normalized_item["region_masks"]
        ):
            raise ValueError(
                "Number of region names must match number of region masks."
            )

        return normalized_item

    def inference(
        self, data: np.ndarray, args: dict | None = None
    ) -> list[dict]:
        """
        Default Image2MultiRegionSegmentation inference method.
        """
        raise NotImplementedError(
            "The inference method is not implemented. Please implement the "
            "inference method in the algorithm Runner class."
        )
