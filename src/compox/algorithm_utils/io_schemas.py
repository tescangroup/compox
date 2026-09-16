"""
Copyright 2024 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
    model_validator,
)
from typing import List, Optional
import numpy as np


class DataSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class GenericSchema(DataSchema):
    data: np.ndarray


class SegmentationSchema(DataSchema):
    mask: np.ndarray


class MultiRegionSegmentationSchema(DataSchema):
    region_names: List[str]
    region_masks: List[np.ndarray]

    @model_validator(mode="after")
    def check_region_names(self):
        if len(self.region_names) != len(self.region_masks):
            raise ValueError(
                "Number of region names must match number of region masks."
            )

        reference_shape = None
        for i, mask in enumerate(self.region_masks):
            if not isinstance(mask, np.ndarray):
                raise ValueError(
                    f"Region mask at index {i} must be a numpy array."
                )
            if mask.ndim != 2:
                raise ValueError(
                    f"Region mask at index {i} must be a 2D numpy array."
                )
            if reference_shape is None:
                reference_shape = mask.shape
            elif mask.shape != reference_shape:
                raise ValueError(
                    "All region masks for one output item must have the same "
                    "shape."
                )
        return self


class ImageSchema(DataSchema):
    image: np.ndarray
    mask: Optional[np.ndarray] = Field(default=None)
    points: Optional[List[np.ndarray]] = Field(default=None)

    @field_validator("image")
    @classmethod
    def check_image(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 and (v.ndim != 3 or v.shape[0] != 3):
            raise ValueError(
                f"Image must be 2D (got {v.ndim} dimensions), grayscale or RGB (got {1 if v.ndim == 2 else v.shape[0]} channels)."
            )
        if v.dtype not in [
            np.uint8,
            np.uint16,
            np.float64,
            np.float32,
            np.float16,
        ]:
            raise ValueError(
                f"Image must be uint8, uint16, float64, float32 or float16 (got {v.dtype})."
            )
        return v


class VolumeSchema(DataSchema):
    volume: np.ndarray
    mask: Optional[np.ndarray] = Field(default=None)
    points: Optional[List[np.ndarray]] = Field(default=None)
    stats: Optional[str] = Field(default=None)

    @field_validator("volume")
    @classmethod
    def check_volume(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 3:
            raise ValueError("Volume must be a 3 dimensional array.")
        return v


class MeshSchema(DataSchema):
    vertices: np.ndarray
    faces: np.ndarray
    stats: Optional[str] = Field(default=None)

    @field_validator("vertices")
    @classmethod
    def check_verts(cls, v: np.ndarray):
        if v.ndim != 2 or v.shape[1] != 3 or v.dtype != float:
            raise ValueError("Mesh vertices must be Nx3 numpy array of floats")
        return v

    @field_validator("faces")
    @classmethod
    def check_faces(cls, v: np.ndarray):
        if v.ndim != 2 or v.shape[1] != 3 or v.dtype != int:
            raise ValueError(
                "Mesh vertices must be Nx3 numpy array of integers pointing into vertices array."
            )
        return v


class AlignmentSchema(DataSchema):
    points1: Optional[List[np.ndarray]] = Field(default=[])
    points2: Optional[List[np.ndarray]] = Field(default=[])
    confidence: Optional[List[float]] = Field(default=[])
    transform_matrix: Optional[np.ndarray] = Field(default=None)
    translation_matrix: Optional[np.ndarray] = Field(default=None)


class EmbeddingSchema(DataSchema):
    features: np.ndarray
    input_size: tuple
    original_size: tuple
