"""
Copyright 2025 TESCAN 3DIM, s.r.o.
All rights reserved
"""

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)
from typing import List, Literal, Union
import warnings
import re


class ParameterConfigSchema(BaseModel):
    type: Literal[
        "string",
        "int",
        "float",
        "bool",
        "int_range",
        "float_range",
        "string_enum",
        "int_enum",
        "float_enum",
        "string_list",
        "int_list",
        "float_list",
        "bool_list",
    ]
    default: Union[
        str,
        int,
        float,
        bool,
        List[str],
        List[int],
        List[float],
        List[bool],
        None,
    ] = None
    adjustable: bool = False
    options: List[
        Union[str, int, float, List[str], List[int], List[float], List[bool]]
    ] = Field(
        default_factory=list
    )  # For enum and list types
    min: Union[int, float, None] = None  # For range types
    max: Union[int, float, None] = None  # For range types
    step: Union[int, float, None] = None  # For range types
    decimal_precision: int | None = None


class AdditionalParameterSchema(BaseModel):
    name: str
    displayed_name: str = ""
    description: str = ""
    config: ParameterConfigSchema

    @staticmethod
    def _default_displayed_name(name: str) -> str:
        """
        Convert a machine-friendly parameter name to a human-friendly label.
        """
        normalized = name.replace("_", " ").replace("-", " ").strip()
        normalized = re.sub(r"\s+", " ", normalized)
        if not normalized:
            return name
        return normalized[:1].upper() + normalized[1:]

    @model_validator(mode="after")
    def check_default_type(self):
        if self.displayed_name == "":
            self.displayed_name = self._default_displayed_name(self.name)
        if self.description == "":
            warnings.warn(
                f"Description is empty for additional parameter '{self.name}'. "
                "Consider providing a description for better documentation."
            )
        if self.config.default is None:
            warnings.warn(
                f"Default value is None for additional parameter '{self.name}'. "
                "Consider providing a default value."
            )
        expected_types = {
            "string": str,
            "int": int,
            "float": float,
            "bool": bool,
            "int_range": int,
            "float_range": float,
            "string_enum": str,
            "int_enum": int,
            "float_enum": float,
            "string_list": list,
            "int_list": list,
            "float_list": list,
            "bool_list": list,
        }

        expected_type = expected_types.get(self.config.type)
        float_types = {"float", "float_range", "float_enum", "float_list"}
        if self.config.decimal_precision is not None:
            if self.config.decimal_precision < 0:
                raise ValueError(
                    f"Validation error in additional parameter {self.name}. "
                    "decimal_precision must be greater than or equal to 0."
                )
            if self.config.type not in float_types:
                raise ValueError(
                    f"Validation error in additional parameter {self.name}. "
                    "decimal_precision can only be used with float parameter types."
                )
        # this checks if the default value matches the expected type
        if (
            not isinstance(self.config.default, expected_type)
            and self.config.default is not None
        ):
            raise ValueError(
                f"Validation error in additional parameter {self.name}. "
                f"Default value type mismatch: Expected {expected_type.__name__} "
                f"for type '{self.config.type}', but got {type(self.config.default).__name__}."
            )
        # this checks that range types have min and max values that are of the same type as the default value,
        # are not None, and that min is less than max
        if self.config.type in ["int_range", "float_range"]:
            if (
                self.config.min is None
                or self.config.max is None
                or self.config.step is None
            ):
                raise ValueError(
                    f"Validation error in additional parameter {self.name}. "
                    f"Min, max, and step values must be provided for range type '{self.config.type}'."
                )
            if (
                not isinstance(self.config.min, expected_type)
                or not isinstance(self.config.max, expected_type)
                or not isinstance(self.config.step, expected_type)
            ):
                raise ValueError(
                    f"Validation error in additional parameter {self.name}. "
                    f"Min, max, and step values must be of type {expected_type.__name__} "
                    f"for range type '{self.config.type}'."
                )
            if self.config.min >= self.config.max:
                raise ValueError(
                    f"Validation error in additional parameter {self.name}. "
                    f"Min value must be less than max value for range type '{self.config.type}'."
                )
        # this checks that enum types have options that are of the same type as the default value
        # and that the options are not empty
        if self.config.type in ["string_enum", "int_enum", "float_enum"]:
            if not self.config.options:
                raise ValueError(
                    f"Validation error in additional parameter {self.name}. "
                    f"Options must be provided for enum type '{self.config.type}'."
                )
            if not all(
                isinstance(option, expected_type)
                for option in self.config.options
            ):
                raise ValueError(
                    f"Validation error in additional parameter {self.name}. "
                    f"All options must be of type {expected_type.__name__} "
                    f"for enum type '{self.config.type}'."
                )
            if (
                self.config.default not in self.config.options
                and self.config.default is not None
            ):
                raise ValueError(
                    f"Validation error in additional parameter {self.name}. "
                    f"Default value '{self.config.default}' must be one of the options for "
                    f"enum type '{self.config.type}'. The options are: {self.config.options}."
                )
            # is default is None, assign the first option as default
            if self.config.default is None:
                self.config.default = self.config.options[0]
        # this checks that list types have options that are of the same type as the default value
        # and that the options are not empty
        if self.config.type in [
            "string_list",
            "int_list",
            "float_list",
            "bool_list",
        ]:
            if not self.config.options:
                warnings.warn(
                    f"Options are empty for additional parameter {self.name}. "
                    f"This may or may not be intentional. "
                    f"Consider providing options for list type '{self.config.type}'."
                )
            if not all(
                isinstance(option, expected_type)
                for option in self.config.options
            ):
                raise ValueError(
                    f"Validation error in additional parameter {self.name}. "
                    f"All options must be of type {expected_type.__name__} "
                    f"for list type '{self.config.type}'."
                )

        return self


class AlgorithmConfigSchema(BaseModel):
    algorithm_type: Literal[
        "Image2Alignment",
        "Image2Image",
        "Image2Segmentation",
        "Image2MultiRegionSegmentation",
        "Image2Embedding",
        "Undefined",
        "Generic",
    ] = "Undefined"
    tags: List[str] = Field(default_factory=list)
    description: str = ""
    supported_devices: List[Literal["cpu", "gpu", "mps"]] = ["cpu"]
    default_device: Literal["cpu", "gpu", "", "mps"] = "cpu"
    additional_parameters: List[AdditionalParameterSchema] = Field(
        default_factory=list
    )
    training_parameters: List[AdditionalParameterSchema] = Field(
        default_factory=list
    )
    benchmark_outputs: List[AdditionalParameterSchema] = Field(
        default_factory=list
    )
    removable: bool = False
    exportable: bool = True

    @model_validator(mode="after")
    def check_algorithm_type(self):

        if self.algorithm_type == "Undefined":
            warnings.warn(
                "Algorithm type is set to 'Undefined'. "
                "Consider specifying a valid algorithm type. "
                "Use 'Generic' for generic algorithms."
            )
        if self.tags == []:
            warnings.warn(
                "Tags are empty. "
                "Consider providing tags for better categorization. "
                "Tags are used to filter algorithms in the REST API."
            )
        if self.description == "":
            warnings.warn(
                "Description is empty. "
                "Consider providing a description for better documentation."
            )
        if self.supported_devices == []:
            warnings.warn(
                "Supported devices are empty. "
                "Consider providing supported devices for better compatibility. "
                "You can use 'cpu', 'gpu', or both."
            )
        if self.default_device == "":
            warnings.warn(
                "Default device is empty. "
                "Consider providing a default device for better compatibility. "
                "You can use 'cpu', 'gpu'."
            )
        # Check if the default device is valid
        if self.default_device not in self.supported_devices:
            raise ValueError(
                f"Default device '{self.default_device}' is not in the list of supported devices "
                f"{self.supported_devices}."
            )
        return self


if __name__ == "__main__":
    import toml
    import os

    # Load the TOML file
    toml_file_path = "C:/Users/Jan Matula/Work/python-computing-backend/compox/algorithms/template_segmentation_algorithm/pyproject.toml"
    if os.path.exists(toml_file_path):
        with open(toml_file_path, "r") as f:
            toml_data = toml.load(f)
    else:
        print(f"TOML file not found at {toml_file_path}.")
        toml_data = {}

    # Extract the algorithm configuration from the TOML data
    algorithm_config = toml_data.get("tool", {}).get("compox", {})

    if algorithm_config:
        # Create an instance of AlgorithmConfigSchema using the extracted data
        algorithm_config_instance = AlgorithmConfigSchema(**algorithm_config)

        print(algorithm_config_instance)
