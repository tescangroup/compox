"""
Copyright 2025 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from __future__ import annotations
from typing import Any, Dict, List

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
)


class CheckpointManifest(BaseModel):
    model_config = ConfigDict(
        extra="ignore",  # ignore unknown top-level keys
        populate_by_name=True,
        str_max_length=10_000,
        json_schema_extra={
            "example": {
                "checkpoint_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
                "parent_algorithm_id": "a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6",
                "training_id": "7g8h9i0j-k1l2m3n4o5p6-a1b2c3d4-5e6f",
                "assets": {
                    "model.pt": "8aef7c9e-1c2d-3e4f-5g6h-7i8j9k0l1m2n",
                },
                "created_at": "2025-09-02T11:42:00Z",
                "properties": {
                    "epoch": 10,
                    "val_loss": 0.0234,
                    "train_accuracy": 0.9876,
                    "classification": "best",
                },
                "tags": ["run:experiment-42", "author:jan"],
                "parent_checkpoint_id": "previous-checkpoint-uuid-if-any",
            }
        },
    )

    # Fields
    checkpoint_id: str = Field(..., description="Checkpoint identifier (UUID).")
    parent_algorithm_id: str = Field(
        ..., description="Parent algorithm identifier (UUID)."
    )
    parent_algorithm_key: str | None = Field(
        default=None,
        description=(
            "Parent algorithm storage key in algorithm-store. When present, "
            "checkpoint registration can update the parent record without "
            "rescanning the algorithm store."
        ),
    )
    training_id: str = Field(..., description="Training run identifier (UUID).")
    assets: Dict[str, str] = Field(
        ...,
        description=(
            "Dictionary mapping asset filenames to their corresponding file IDs."
        ),
    )
    created_at: str = Field(
        ...,
        description=(
            "Timestamp of when the checkpoint was created in ISO 8601 format."
        ),
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Dictionary of arbitrary properties associated with the checkpoint. "
            "Can include metrics like 'epoch', 'val_loss', etc."
        ),
    )

    tags: List[str] = Field(
        default_factory=list,
        description=(
            "List of tags associated with the checkpoint. Tags can be defined either "
            "as a list of strings (e.g. ['best', 'experiment-42']) or as a key-value pair "
            "separated by colon (e.g. ['run:experiment-42', 'author:jan'])."
        ),
    )
    parent_checkpoint_id: str | None = Field(
        default=None,
        description=(
            "The id of the parent checkpoint, if any. This can be used to track "
            "the lineage of checkpoints."
        ),
    )

    @field_validator("tags", mode="before")
    def normalize_tags(cls, v):
        """Ensure tags are a list of strings."""
        if isinstance(v, str):
            return [v]
        elif isinstance(v, list):
            return [str(tag) for tag in v]
        else:
            raise ValueError("Tags must be a string or a list of strings.")


if __name__ == "__main__":
    # Example usage
    example = CheckpointManifest.model_validate(
        {
            "checkpoint_id": "3f8b2b65-1c3d-4a55-8a1d-7c2d4c0b9a90",
            "parent_algorithm_id": "a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6",
            "training_id": "7g8h9i0j-k1l2m3n4o5p6-a1b2c3d4-5e6f",
            "assets": {
                "model.pt": "8aef7c9e-1c2d-3e4f-5g6h-7i8j9k0l1m2n",
            },
            "created_at": "2025-09-02T11:42:00Z",
            "properties": {
                "epoch": 10,
                "val_loss": 0.0234,
                "train_accuracy": 0.9876,
                "classification": "best",
            },
            "tags": ["run:experiment-42", "author:jan"],
            "parent_checkpoint_id": "previous-checkpoint-uuid-if-any",
        }
    )
    print(example.model_dump_json(indent=4))
