"""
 Copyright 2026 TESCAN GROUP, a.s.
All rights reserved
"""

import numpy as np
from compox.algorithm_utils.Image2MultiRegionSegmentationRunner import (
    Image2MultiRegionSegmentationRunner,
)
from image_segmentation.segmentation_utils import threshold_image_multiregion


class Runner(Image2MultiRegionSegmentationRunner):
    """
    The runner class for the multi-region segmentation algorithm.
    """

    def load_assets(self):
        """
        Here you can load the assets needed for the algorithm. This can be
        the model, the weights, etc. The assets are loaded upon the first
        call of the algorithm and are cached with the algorithm instance.
        """
        pass

    def inference(
        self, data: np.ndarray, args: dict | None = None
    ) -> list[dict]:
        """
        Run the inference.

        Parameters
        ----------
        data : np.ndarray
            The images to be segmented.
        args : dict
            The arguments for the algorithm.

        Returns
        -------
        list[dict]
            One multi-region segmentation record per input image.
        """
        args = args or {}
        thresholding_algorithm = args.get("thresholding_algorithm", "otsu")
        low_region_scale = float(args.get("low_region_scale", 0.8))
        high_region_scale = float(args.get("high_region_scale", 1.2))

        self.log_message(
            "Starting inference with thresholding algorithm: "
            f"{thresholding_algorithm}"
        )

        output = threshold_image_multiregion(
            data,
            thresholding_algorithm=thresholding_algorithm,
            low_region_scale=low_region_scale,
            high_region_scale=high_region_scale,
        )
        mid_masks, high_masks = output

        output = []
        for mid_mask, high_mask in zip(mid_masks, high_masks):
            output.append(
                {
                    "mid_intensity": mid_mask,
                    "high_intensity": high_mask,
                }
            )

        self.set_progress(0.5)

        return output
