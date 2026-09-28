"""Core spectral-imaging dataset containers."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from pycardinal.core.metadata import MassDataFrame, PositionDataFrame
from pycardinal.core.spectra_arrays import SpectraArrays


def _position_data(
    pixel_data: PositionDataFrame | pd.DataFrame | None,
    count: int,
) -> PositionDataFrame:
    if pixel_data is None:
        if count:
            raise ValueError("pixel_data is required when spectra contain pixels")
        return PositionDataFrame(np.empty((0, 2)))
    result = PositionDataFrame(pixel_data)
    if len(result) != count:
        raise ValueError(f"pixel_data has {len(result)} rows; expected {count}")
    return result


def _copy_value(value: Any, deep: bool) -> Any:
    if not deep:
        return value
    if sparse.issparse(value):
        return cast(Any, value).copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_copy_value(item, True) for item in value]
    return copy.deepcopy(value)


class SpectralImagingData:
    """Shared base for spectral-imaging datasets."""

    _feature_data: pd.DataFrame | None
    experiment_data: dict[str, Any] | None

    def __init__(
        self,
        spectra_data: SpectraArrays | Mapping[str, Any] | None = None,
        pixel_data: PositionDataFrame | pd.DataFrame | None = None,
        *,
        processing: list[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._spectra_data = (
            spectra_data.copy()
            if isinstance(spectra_data, SpectraArrays)
            else SpectraArrays(spectra_data)
        )
        self._pixel_data = _position_data(pixel_data, self._infer_pixel_count())
        self._processing = list(processing or [])
        self.metadata = dict(metadata or {})
        self._feature_data = None
        self.experiment_data = None

    def _infer_pixel_count(self) -> int:
        if not self._spectra_data:
            return 0
        value = next(iter(self._spectra_data.values()))
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) == 2:
            return int(shape[1])
        return len(value)

    @property
    def spectra_data(self) -> SpectraArrays:
        """Named spectral arrays stored by the dataset."""
        return self._spectra_data

    @property
    def pixel_data(self) -> PositionDataFrame:
        """Metadata with one row per pixel/spectrum."""
        return self._pixel_data

    @pixel_data.setter
    def pixel_data(self, value: PositionDataFrame | pd.DataFrame) -> None:
        """Replace pixel metadata after validating the pixel count."""
        self._pixel_data = _position_data(value, self._infer_pixel_count())

    @property
    def coord(self) -> pd.DataFrame:
        """Pixel coordinate columns."""
        return self._pixel_data.coord

    @property
    def run(self) -> pd.Categorical:
        """Run labels for each pixel."""
        return self._pixel_data.run

    @property
    def processing(self) -> list[Any]:
        """Queued processing steps, in execution order."""
        return self._processing

    @processing.setter
    def processing(self, value: list[Any]) -> None:
        """Replace the deferred processing queue."""
        self._processing = list(value)

    def __len__(self) -> int:
        """Number of pixels/spectra in the dataset."""
        return len(self._pixel_data)

    def copy(self, deep: bool = True) -> SpectralImagingData:
        """Copy the dataset, optionally copying underlying array values."""
        result = copy.copy(self)
        result._spectra_data = SpectraArrays(
            {
                name: _copy_value(value, deep)
                for name, value in self._spectra_data.items()
            }
        )
        result._pixel_data = PositionDataFrame(self._pixel_data.copy())
        if self._feature_data is not None:
            feature_data = self._feature_data.copy(deep=deep)
            result._feature_data = (
                MassDataFrame(feature_data)
                if isinstance(self._feature_data, MassDataFrame)
                else pd.DataFrame(feature_data)
            )
        result._processing = list(self._processing)
        result.metadata = copy.deepcopy(self.metadata) if deep else dict(self.metadata)
        if self.experiment_data is not None:
            result.experiment_data = (
                copy.deepcopy(self.experiment_data) if deep else self.experiment_data
            )
        return result


class SpectralImagingArrays(SpectralImagingData):
    """Dataset of per-pixel spectra whose m/z axes may differ."""


class MSImagingArrays(SpectralImagingArrays):
    """Mass-spectrometry dataset stored as ragged m/z/intensity spectra."""

    def __init__(
        self,
        spectra_data: SpectraArrays | Mapping[str, Any] | None = None,
        *,
        mz: Sequence[ArrayLike] | None = None,
        intensity: Sequence[ArrayLike] | None = None,
        pixel_data: PositionDataFrame | pd.DataFrame | None = None,
        experiment_data: Mapping[str, Any] | None = None,
        centroided: bool | None = None,
        continuous: bool = False,
        processing: list[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if spectra_data is None:
            if mz is None or intensity is None:
                spectra_data = SpectraArrays()
            else:
                spectra_data = SpectraArrays(
                    {"mz": list(mz), "intensity": list(intensity)}
                )
        super().__init__(
            spectra_data,
            pixel_data,
            processing=processing,
            metadata=metadata,
        )
        if self._spectra_data:
            if "mz" not in self._spectra_data or "intensity" not in self._spectra_data:
                raise ValueError("MSImagingArrays requires 'mz' and 'intensity' arrays")
            if len(self._spectra_data["mz"]) != len(self._spectra_data["intensity"]):
                raise ValueError(
                    "mz and intensity must contain the same number of spectra"
                )
            for pixel, (mass, signal) in enumerate(
                zip(self._spectra_data["mz"], self._spectra_data["intensity"])
            ):
                if np.asarray(mass).ndim != 1 or np.asarray(signal).ndim != 1:
                    raise ValueError(
                        f"spectrum {pixel} m/z and intensity must be one-dimensional"
                    )
                if len(mass) != len(signal):
                    raise ValueError(
                        f"spectrum {pixel} m/z and intensity lengths differ"
                    )
        self._pixel_data = _position_data(pixel_data, self._infer_pixel_count())
        self.experiment_data = dict(experiment_data or {})
        self.centroided = centroided
        self.continuous = bool(continuous)

    @property
    def mz(self) -> list[Any]:
        """Per-pixel m/z arrays."""
        return cast(list[Any], self._spectra_data["mz"])

    @mz.setter
    def mz(self, values: Sequence[ArrayLike]) -> None:
        """Replace ragged per-pixel m/z arrays."""
        self._set_ragged_data("mz", values)

    @property
    def intensity(self) -> list[Any]:
        """Per-pixel intensity arrays."""
        return cast(list[Any], self._spectra_data["intensity"])

    @intensity.setter
    def intensity(self, values: Sequence[ArrayLike]) -> None:
        """Replace ragged per-pixel intensity arrays."""
        self._set_ragged_data("intensity", values)

    def _set_ragged_data(self, name: str, values: Sequence[ArrayLike]) -> None:
        if len(values) != len(self):
            raise ValueError(f"{name} must contain one array per pixel")
        other_name = "intensity" if name == "mz" else "mz"
        if other_name in self._spectra_data:
            other_values = self._spectra_data[other_name]
            for pixel, (value, other) in enumerate(zip(values, other_values)):
                value_array = np.asarray(value)
                other_array = np.asarray(other)
                if value_array.ndim != 1 or other_array.ndim != 1:
                    raise ValueError("each spectrum must be one-dimensional")
                if len(value_array) != len(other_array):
                    raise ValueError(
                        f"m/z and intensity lengths differ at pixel {pixel}"
                    )
        self._spectra_data[name] = list(values)


class SpectralImagingExperiment(SpectralImagingData):
    """Shared-domain data matrix with features in rows and pixels in columns."""

    def __init__(
        self,
        spectra_data: SpectraArrays | Mapping[str, Any] | ArrayLike | None = None,
        *,
        feature_data: pd.DataFrame | None = None,
        pixel_data: PositionDataFrame | pd.DataFrame | None = None,
        processing: list[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if spectra_data is None:
            arrays = SpectraArrays()
        elif isinstance(spectra_data, SpectraArrays):
            arrays = spectra_data.copy()
        elif isinstance(spectra_data, Mapping):
            arrays = SpectraArrays(spectra_data)
        else:
            matrix = np.asarray(spectra_data)
            arrays = SpectraArrays({"intensity": matrix})
        if arrays:
            first = next(iter(arrays.values()))
            if len(first.shape) != 2:
                raise ValueError("shared-domain spectra must be two-dimensional")
            n_features, n_pixels = first.shape
        else:
            n_features, n_pixels = 0, 0
        super().__init__(arrays, pixel_data, processing=processing, metadata=metadata)
        if feature_data is None:
            feature_data = pd.DataFrame(index=range(n_features))
        if len(feature_data) != n_features:
            raise ValueError(
                f"feature_data has {len(feature_data)} rows; expected {n_features}"
            )
        self._feature_data = pd.DataFrame(feature_data).reset_index(drop=True)
        if len(self._pixel_data) != n_pixels:
            raise ValueError(
                f"pixel_data has {len(self._pixel_data)} rows; expected {n_pixels}"
            )

    @property
    def feature_data(self) -> pd.DataFrame:
        """Metadata with one row per shared-domain feature."""
        if self._feature_data is None:
            raise RuntimeError("feature metadata is not initialized")
        return self._feature_data

    @feature_data.setter
    def feature_data(self, value: pd.DataFrame) -> None:
        """Replace feature metadata after validating the feature count."""
        if len(value) != self.shape[0]:
            raise ValueError(f"feature_data must have {self.shape[0]} rows")
        self._feature_data = pd.DataFrame(value).reset_index(drop=True)

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape as ``(features, pixels)``."""
        if not self._spectra_data:
            return (0, 0)
        return tuple(next(iter(self._spectra_data.values())).shape)

    def spectra(self, name: str = "intensity") -> Any:
        """Return a named shared-domain matrix (features by pixels)."""
        return self._spectra_data[name]

    def __getitem__(self, key: tuple[Any, Any]) -> SpectralImagingExperiment:
        """Subset by ``dataset[feature_selector, pixel_selector]``."""
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("index shared-domain data as dataset[features, pixels]")
        feature_selector, pixel_selector = key
        row_indices = np.atleast_1d(np.arange(self.shape[0])[feature_selector])
        column_indices = np.atleast_1d(np.arange(self.shape[1])[pixel_selector])
        arrays = {}
        for name, values in self._spectra_data.items():
            if sparse.issparse(values):
                sparse_values = cast(Any, values)
                arrays[name] = sparse_values[row_indices, :][:, column_indices]
            else:
                arrays[name] = np.asarray(values)[np.ix_(row_indices, column_indices)]
        feature_data = self.feature_data.iloc[row_indices].reset_index(drop=True)
        pixel_data = PositionDataFrame(
            self._pixel_data.iloc[column_indices].reset_index(drop=True)
        )
        return type(self)(
            arrays,
            feature_data=feature_data,
            pixel_data=pixel_data,
            processing=self.processing,
            metadata=self.metadata,
        )


class MSImagingExperiment(SpectralImagingExperiment):
    """Shared-m/z mass-spectrometry imaging experiment.

    The intensity matrix is always oriented ``(features, pixels)``; each row
    corresponds to the matching sorted m/z value in ``feature_data``.
    """

    def __init__(
        self,
        spectra_data: SpectraArrays | Mapping[str, Any] | ArrayLike | None = None,
        *,
        feature_data: MassDataFrame | pd.DataFrame | None = None,
        pixel_data: PositionDataFrame | pd.DataFrame | None = None,
        experiment_data: Mapping[str, Any] | None = None,
        centroided: bool | None = None,
        processing: list[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(feature_data, MassDataFrame):
            mass_data = MassDataFrame(feature_data)
        elif feature_data is not None:
            mass_data = MassDataFrame(feature_data)
        else:
            mass_data = None
        super().__init__(
            spectra_data,
            feature_data=mass_data,
            pixel_data=pixel_data,
            processing=processing,
            metadata=metadata,
        )
        if "intensity" not in self._spectra_data:
            raise ValueError("MSImagingExperiment requires an 'intensity' matrix")
        if mass_data is None:
            raise ValueError(
                "MSImagingExperiment requires feature_data with m/z values"
            )
        if "mz" not in mass_data:
            raise ValueError("feature_data must include m/z values")
        self._feature_data = mass_data
        self.experiment_data = dict(experiment_data or {})
        self.centroided = centroided

    @property
    def mz(self) -> NDArray[np.float64]:
        """Shared m/z values, one per matrix row."""
        if not isinstance(self._feature_data, MassDataFrame):
            raise RuntimeError("MSImagingExperiment feature_data is not mass metadata")
        return self._feature_data.mz

    @mz.setter
    def mz(self, values: ArrayLike) -> None:
        """Replace shared m/z values after validating the feature count."""
        mass_values = np.asarray(values)
        if mass_values.ndim != 1 or len(mass_values) != self.shape[0]:
            raise ValueError(f"mz must contain {self.shape[0]} values")
        if not isinstance(self._feature_data, MassDataFrame):
            raise RuntimeError("MSImagingExperiment feature_data is not mass metadata")
        feature_data = pd.DataFrame(self._feature_data).copy()
        feature_data["mz"] = cast(Any, mass_values)
        self._feature_data = MassDataFrame(feature_data)

    @property
    def intensity(self) -> Any:
        """Intensity matrix oriented as ``(features, pixels)``."""
        return self._spectra_data["intensity"]

    @intensity.setter
    def intensity(self, value: Any) -> None:
        """Replace the shared intensity matrix without changing its shape."""
        matrix = value if sparse.issparse(value) else np.asarray(value)
        matrix_shape = cast(Any, matrix).shape
        if len(matrix_shape) != 2 or matrix_shape != self.shape:
            raise ValueError(f"intensity must have shape {self.shape}")
        self._spectra_data["intensity"] = matrix

    def is_centroided(self) -> bool:
        """Return whether the data is known to be centroided."""
        return self.centroided is True

    def __getitem__(self, key: tuple[Any, Any]) -> MSImagingExperiment:
        """Subset by feature rows and pixel columns."""
        return cast(MSImagingExperiment, super().__getitem__(key))
