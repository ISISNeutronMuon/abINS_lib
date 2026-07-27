"""I/O and serialization utilities for abinslib data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, Self
import warnings

from packaging.version import parse as parse_version

from .util import get_version


class JSONMixin(ABC):
    """Mixin providing JSON serialization and deserialization methods."""

    @abstractmethod
    def to_dict(self) -> Mapping[str, Any]:
        """Convert object to a dictionary."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data_dict: Any) -> Self:
        """Instantiate object from a dictionary."""

    def to_json(self) -> str:
        """Convert object representation to a JSON string."""
        data_dict = dict(self.to_dict())
        data_dict["__abinslib_class__"] = type(self).__name__
        data_dict["__abinslib_version__"] = get_version()
        return json.dumps(data_dict, indent=4, sort_keys=True)

    @staticmethod
    def _validate_class(file_class: str, expected_class: str) -> None:
        """Ensure JSON class metadata matches expected target class name."""
        if file_class != expected_class:
            raise ValueError(
                f"JSON data class '{file_class}' does not match "
                f"expected class '{expected_class}'."
            )

    @staticmethod
    def _validate_version(file_version: str) -> None:
        """Issue warnings if JSON version is DEVELOPMENT or newer than installed."""
        current_version = get_version()

        if current_version == "DEVELOPMENT":
            return

        if file_version == "DEVELOPMENT":
            warnings.warn(
                "JSON data was generated with a DEVELOPMENT version of abinslib.",
                UserWarning,
                stacklevel=3,
            )
            return

        if parse_version(file_version) > parse_version(current_version):
            warnings.warn(
                f"JSON data version ({file_version}) is newer "
                f"than current abinslib version ({current_version}).",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Instantiate object from a JSON string."""
        data_dict = json.loads(json_str)

        try:
            file_class = data_dict.pop("__abinslib_class__")
        except KeyError:
            raise ValueError("JSON data is missing required '__abinslib_class__' key.")
        cls._validate_class(file_class, cls.__name__)

        try:
            file_version = data_dict.pop("__abinslib_version__")
        except KeyError:
            raise ValueError(
                "JSON data is missing required '__abinslib_version__' key."
            )
        cls._validate_version(file_version)

        return cls.from_dict(data_dict)

    def to_json_file(self, filename: Path | str) -> None:
        """Write object representation to a JSON file."""
        Path(filename).write_text(self.to_json())

    @classmethod
    def from_json_file(cls, filename: Path | str) -> Self:
        """Read object representation from a JSON file."""
        return cls.from_json(Path(filename).read_text())
