"""Summary statistics for shared-domain spectral-imaging data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from pycardinal.core.imaging_data import MSImagingExperiment, SpectralImagingExperiment

_SUPPORTED_STATS = {
    "min",
    "max",
    "prod",
    "sum",
    "mean",
    "var",
    "sd",
    "any",
    "all",
    "nnzero",
}


def _matrix(value: Any) -> NDArray[np.float64]:
    if sparse.issparse(value):
        value = getattr(value, "toarray")()
    elif hasattr(value, "to_numpy"):
        value = value.to_numpy()
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("summary input must be a two-dimensional matrix")
    return matrix


def _values(value: Any, na_rm: bool) -> NDArray[np.float64]:
    values = np.asarray(value, dtype=float).ravel()
    if na_rm:
        values = values[~np.isnan(values)]
    return values


def _validate_stat(stat: str) -> str:
    if stat not in _SUPPORTED_STATS:
        options = ", ".join(sorted(_SUPPORTED_STATS))
        raise ValueError(f"unsupported statistic {stat!r}; expected one of {options}")
    return stat


def _reduce(values: Any, stat: str, na_rm: bool) -> float | bool | int:
    stat = _validate_stat(stat)
    vector = _values(values, na_rm)
    if vector.size == 0:
        if stat in {"any", "all"}:
            return stat == "all"
        return float("nan")
    if stat == "min":
        return float(np.min(vector))
    if stat == "max":
        return float(np.max(vector))
    if stat == "prod":
        return float(np.prod(vector))
    if stat == "sum":
        return float(np.sum(vector))
    if stat == "mean":
        return float(np.mean(vector))
    if stat == "var":
        return float(np.var(vector, ddof=1)) if vector.size > 1 else float("nan")
    if stat == "sd":
        return float(np.std(vector, ddof=1)) if vector.size > 1 else float("nan")
    if stat == "any":
        return bool(np.any(vector != 0))
    if stat == "all":
        return bool(np.all(vector != 0))
    return int(np.count_nonzero(vector))


def _reduce_axis(matrix: Any, stat: str, axis: int, na_rm: bool) -> NDArray[Any]:
    values = _matrix(matrix)
    if axis == 1:
        rows = values
    elif axis == 0:
        rows = values.T
    else:
        raise ValueError("axis must be 0 or 1")
    return np.asarray([_reduce(row, stat, na_rm) for row in rows])


def row_stats(
    x: Any,
    stat: str,
    *,
    na_rm: bool = False,
) -> NDArray[Any]:
    """Reduce each row of a two-dimensional dense or sparse matrix."""
    return _reduce_axis(x, stat, axis=1, na_rm=na_rm)


def col_stats(
    x: Any,
    stat: str,
    *,
    na_rm: bool = False,
) -> NDArray[Any]:
    """Reduce each column of a two-dimensional dense or sparse matrix."""
    return _reduce_axis(x, stat, axis=0, na_rm=na_rm)


def _stat_map(
    stat: str | Mapping[str, str],
    default_name: str,
) -> dict[str, str]:
    if isinstance(stat, str):
        return {default_name: _validate_stat(stat)}
    result = {str(name): _validate_stat(value) for name, value in stat.items()}
    if not result:
        raise ValueError("stat must contain at least one statistic")
    return result


def _group_masks(
    groups: ArrayLike | None, length: int
) -> list[tuple[str, NDArray[np.bool_]]]:
    if groups is None:
        return []
    labels = np.asarray(groups)
    if labels.ndim != 1 or len(labels) != length:
        raise ValueError(f"groups must contain one label per entry; expected {length}")
    if pd.isna(labels).any():
        raise ValueError("groups cannot contain missing labels")
    return [(str(label), labels == label) for label in pd.unique(labels)]


def summarize_features(
    obj: SpectralImagingExperiment,
    stat: str | Mapping[str, str] = "mean",
    groups: ArrayLike | None = None,
    *,
    na_rm: bool = False,
) -> SpectralImagingExperiment:
    """Add per-feature summaries across pixels to feature metadata.

    Group labels refer to pixels. Grouped columns are named ``group.stat``.
    """
    if not isinstance(obj, SpectralImagingExperiment):
        raise TypeError("summarize_features requires a shared-domain experiment")
    experiment = obj if isinstance(obj, MSImagingExperiment) else None
    if experiment is None:
        raise TypeError("summarize_features requires a mass-spectrometry experiment")
    matrix = _matrix(experiment.intensity)
    result = cast(MSImagingExperiment, experiment.copy())
    feature_data = result.feature_data.copy()
    stat_map = _stat_map(stat, "mean")
    masks = _group_masks(groups, matrix.shape[1])
    if not masks:
        masks = [("", np.ones(matrix.shape[1], dtype=bool))]
    for group, mask in masks:
        for name, statistic in stat_map.items():
            column = name if group == "" else f"{group}.{name}"
            feature_data[column] = row_stats(matrix[:, mask], statistic, na_rm=na_rm)
    result.feature_data = feature_data
    return result


def summarize_pixels(
    obj: SpectralImagingExperiment,
    stat: str | Mapping[str, str] = {"tic": "sum"},
    groups: ArrayLike | None = None,
    *,
    na_rm: bool = False,
) -> SpectralImagingExperiment:
    """Add per-pixel summaries across features to pixel metadata.

    Group labels refer to features. Grouped columns are named ``group.stat``.
    """
    if not isinstance(obj, SpectralImagingExperiment):
        raise TypeError("summarize_pixels requires a shared-domain experiment")
    experiment = obj if isinstance(obj, MSImagingExperiment) else None
    if experiment is None:
        raise TypeError("summarize_pixels requires a mass-spectrometry experiment")
    matrix = _matrix(experiment.intensity)
    result = cast(MSImagingExperiment, experiment.copy())
    pixel_data = result.pixel_data.copy()
    stat_map = _stat_map(stat, "tic")
    masks = _group_masks(groups, matrix.shape[0])
    if not masks:
        masks = [("", np.ones(matrix.shape[0], dtype=bool))]
    for group, mask in masks:
        for name, statistic in stat_map.items():
            column = name if group == "" else f"{group}.{name}"
            pixel_data[column] = col_stats(matrix[mask, :], statistic, na_rm=na_rm)
    result.pixel_data = pixel_data
    return result


__all__ = [
    "col_stats",
    "row_stats",
    "summarize_features",
    "summarize_pixels",
]
