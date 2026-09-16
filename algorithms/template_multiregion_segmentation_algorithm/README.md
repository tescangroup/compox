# Multi-region segmentation algorithm template

This guide mirrors the simple binary segmentation template, but uses the
`Image2MultiRegionSegmentation` runner contract. The input is still a sequence
of images. The output is now a list of datasets where each dataset contains
multiple named region masks.

The algorithm folder is structured as follows:

```plaintext
template_multiregion_segmentation_algorithm/
    |-- __init__.py
    |-- Runner.py
    |-- pyproject.toml
    |-- image_segmentation/
    |   |-- __init__.py
    |   `-- segmentation_utils.py
    `-- README.md
```

## The pyproject.toml file

The metadata is defined in `pyproject.toml`.

```toml
[project]
name = "template_multiregion_segmentation_algorithm"
version = "1.0.0"
```

For the multi-region example, use the `Image2MultiRegionSegmentation`
algorithm type.

```toml
[tool.compox]
algorithm_type = "Image2MultiRegionSegmentation"
tags = ["image-segmentation", "multi-region-segmentation"]
description = "Performs a simple multi-region segmentation of a 3-D image using threshold bands."
```

This example exposes one thresholding method selector and two scalar parameters
that define how far below and above the base threshold the two region bands
should be placed.

## The helper module

The helper module computes one scalar threshold for the whole input stack and
returns two mask stacks:

- `mid_intensity`: pixels above `base_threshold * low_region_scale`
- `high_intensity`: pixels above `base_threshold * high_region_scale`

The `Runner.inference()` method then assembles the final multi-region output,
so the output contract is visible directly in `Runner.py`. Each input image
produces a dictionary with:

```python
{
    "region_names": ["mid_intensity", "high_intensity"],
    "region_masks": [mid_mask, high_mask],
}
```

## The Runner.py file

The runner inherits from
`compox.algorithm_utils.Image2MultiRegionSegmentationRunner`.

```python
import numpy as np
from compox.algorithm_utils.Image2MultiRegionSegmentationRunner import (
    Image2MultiRegionSegmentationRunner,
)
from image_segmentation.segmentation_utils import threshold_image_multiregion


class Runner(Image2MultiRegionSegmentationRunner):
    def inference(
        self, data: np.ndarray, args: dict | None = None
    ) -> list[dict]:
        mid_masks, high_masks = threshold_image_multiregion(...)
        return [
            {
                "mid_intensity": mid_mask,
                "high_intensity": high_mask,
            }
            for mid_mask, high_mask in zip(mid_masks, high_masks)
        ]
```

The important difference from `Image2SegmentationRunner` is the return type.
Instead of a single stacked mask array, `inference` must return a list of
dictionaries, one per input image.

## Deploying the algorithm

To deploy the finished algorithm, use:

```bash
compox deploy-algorithms --config app_server.yaml --name template_multiregion_segmentation_algorithm
```
