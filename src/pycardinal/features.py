"""Feature and pixel selection utilities."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from pycardinal.core.imaging_data import (
    MSImagingArrays,
    MSImagingExperiment,
    SpectralImagingData,
    SpectralImagingExperiment,
)
from pycardinal.core.metadata import PositionDataFrame

_OPERATORS = {"eq", "ne", "lt", "le", "gt", "ge", "in", "notin"}


def _boolean_mask(values: Any, length: int, label: str) -> NDArray[np.bool_]:
    if isinstance(values, pd.Series):
        values = values.fillna(False).to_numpy()
    mask = np.asarray(values)
    if mask.ndim != 1 or len(mask) != length or mask.dtype.kind != "b":
        raise ValueError(f"{label} must produce one boolean value per metadata row")
    return mask.astype(bool, copy=False)


def _condition_mask(
    frame: pd.DataFrame, name: str, condition: Any
) -> NDArray[np.bool_]:
    column = name
    operation = "eq"
    if name not in frame.columns and "__" in name:
        column, operation = name.rsplit("__", 1)
        if operation not in _OPERATORS:
            raise ValueError(f"unsupported condition operator: {operation}")
    if column not in frame.columns:
        raise KeyError(f"metadata has no column {column!r}")

    values = frame[column]
    if callable(condition):
        matched = condition(values)
    elif operation == "in":
        matched = values.isin(condition)
    elif operation == "notin":
        matched = ~values.isin(condition)
    elif operation == "eq" and isinstance(condition, (list, tuple, set, np.ndarray)):
        matched = values.isin(condition)
    else:
        matched = getattr(values, operation)(condition)
    return _boolean_mask(matched, len(frame), f"condition {name!r}")


def _matching_indices(
    frame: pd.DataFrame, query: str | None, conditions: Mapping[str, Any]
) -> NDArray[np.intp]:
    mask = np.ones(len(frame), dtype=bool)
    if query is not None:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        mask &= _boolean_mask(frame.eval(query, engine="python"), len(frame), "query")
    for name, condition in conditions.items():
        mask &= _condition_mask(frame, name, condition)
    return np.flatnonzero(mask)


def features(
    obj: SpectralImagingExperiment,
    query: str | None = None,
    **conditions: Any,
) -> NDArray[np.intp]:
    """Return zero-based feature indices matching metadata predicates.

    ``query`` is a pandas expression, such as ``"mz >= 500 and quality > 0.8"``.
    Keyword conditions use exact column names for equality, or ``column__op``
    with ``eq``, ``ne``, ``lt``, ``le``, ``gt``, ``ge``, ``in``, or ``notin``.
    A callable condition receives the metadata Series and returns a boolean mask.
    Multiple predicates are combined with logical AND.
    """
    if not isinstance(obj, SpectralImagingExperiment):
        raise TypeError("features requires a shared-domain imaging experiment")
    return _matching_indices(obj.feature_data, query, conditions)


def pixels(
    obj: SpectralImagingData,
    query: str | None = None,
    **conditions: Any,
) -> NDArray[np.intp]:
    """Return zero-based pixel indices matching pixel metadata predicates.

    Query strings and keyword conditions follow :func:`features` semantics.
    """
    if not isinstance(obj, SpectralImagingData):
        raise TypeError("pixels requires a spectral-imaging dataset")
    return _matching_indices(obj.pixel_data, query, conditions)


def subset_features(
    obj: SpectralImagingExperiment,
    query: str | None = None,
    **conditions: Any,
) -> SpectralImagingExperiment:
    """Return a shared-domain experiment with matching feature rows."""
    return obj[features(obj, query, **conditions), :]


def _subset_ragged_pixels(
    obj: MSImagingArrays, indices: NDArray[np.intp]
) -> MSImagingArrays:
    spectra_data = {
        name: [copy.deepcopy(values[index]) for index in indices]
        for name, values in obj.spectra_data.items()
    }
    pixel_data = PositionDataFrame(obj.pixel_data.iloc[indices].reset_index(drop=True))
    return type(obj)(
        spectra_data=spectra_data,
        pixel_data=pixel_data,
        experiment_data=obj.experiment_data,
        centroided=obj.centroided,
        continuous=obj.continuous,
        processing=obj.processing,
        metadata=obj.metadata,
    )


def subset_pixels(
    obj: SpectralImagingData,
    query: str | None = None,
    **conditions: Any,
) -> SpectralImagingData:
    """Return a dataset containing pixels matching metadata predicates."""
    indices = pixels(obj, query, **conditions)
    if isinstance(obj, SpectralImagingExperiment):
        return obj[:, indices]
    if isinstance(obj, MSImagingArrays):
        return _subset_ragged_pixels(obj, indices)
    raise TypeError(f"pixel subsetting is not supported for {type(obj).__name__}")


def subset(
    obj: SpectralImagingData,
    select: Any = None,
    subset: Any = None,
) -> SpectralImagingData:
    """Subset shared-domain data by feature and pixel selectors.

    ``select`` indexes feature rows and ``subset`` indexes pixel columns. For
    ragged ``MSImagingArrays``, only pixel selection is defined.
    """
    if isinstance(obj, SpectralImagingExperiment):
        feature_selector = slice(None) if select is None else select
        pixel_selector = slice(None) if subset is None else subset
        return obj[feature_selector, pixel_selector]
    if isinstance(obj, MSImagingArrays):
        if select is not None:
            raise TypeError("feature selection requires a shared-domain experiment")
        if subset is None:
            return obj.copy()
        indices = np.atleast_1d(np.arange(len(obj))[subset])
        return _subset_ragged_pixels(obj, indices.astype(np.intp, copy=False))
    raise TypeError(f"subsetting is not supported for {type(obj).__name__}")


def _selected_indices(selector: Any, length: int) -> NDArray[np.intp]:
    if isinstance(selector, (int, np.integer)):
        index = int(selector)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("feature index is out of range")
        return np.array([index], dtype=np.intp)
    selected = np.atleast_1d(np.arange(length)[selector])
    return np.asarray(selected, dtype=np.intp)


def _rasterize(values: NDArray[np.float64], coord: pd.DataFrame) -> NDArray[np.float64]:
    x_values = np.sort(coord["x"].unique())
    y_values = np.sort(coord["y"].unique())
    x_lookup = {value: index for index, value in enumerate(x_values)}
    y_lookup = {value: index for index, value in enumerate(y_values)}
    image = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for x, y, value in zip(coord["x"], coord["y"], values):
        row, column = y_lookup[y], x_lookup[x]
        if not np.isnan(image[row, column]):
            raise ValueError(
                "duplicate x/y coordinates within a run cannot be rasterized"
            )
        image[row, column] = value
    return image


def _feature_reference(
    obj: SpectralImagingExperiment,
    i: Any = None,
    mz: Any = None,
    ref: Any = None,
) -> Any:
    if ref is not None:
        return np.asarray(ref, dtype=np.float64).ravel()
    if i is not None:
        if isinstance(i, (int, np.integer)):
            indices = np.array([int(i)], dtype=np.intp)
        else:
            indices = np.asarray(i, dtype=np.intp).ravel()
        if indices.size == 0:
            raise ValueError("reference feature index selection is empty")
        experiment = cast(MSImagingExperiment, obj)
        return np.asarray(experiment.intensity[indices], dtype=np.float64)
    if mz is not None:
        experiment = cast(MSImagingExperiment, obj)
        mz_values = np.asarray(mz, dtype=np.float64).ravel()
        matched: list[int] = []
        for value in mz_values:
            matches = np.flatnonzero(
                np.isclose(experiment.mz, value, rtol=1e-6, atol=0.0)
            )
            if matches.size == 0:
                raise ValueError(f"no feature matches m/z value {value!r}")
            matched.extend(int(index) for index in matches)
        unique = np.unique(np.asarray(matched, dtype=np.intp))
        return np.asarray(experiment.intensity[unique], dtype=np.float64)
    raise ValueError("either i, mz, or ref must be provided")


def _colocalization_scores(
    feature_matrix: NDArray[np.float64],
    reference: NDArray[np.float64],
    threshold: Any,
) -> dict[str, float]:
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(1, -1)
    if reference.ndim != 1:
        reference = reference.ravel()
    if feature_matrix.shape[1] != reference.size:
        raise ValueError(
            "reference length does not match the number of pixels in the experiment"
        )

    ref_values = reference.astype(float, copy=False)
    if callable(threshold):
        cutoff = float(threshold(ref_values))
    elif isinstance(threshold, str):
        lowered = threshold.lower()
        if lowered == "median":
            cutoff = float(np.median(ref_values))
        else:
            raise ValueError(f"unsupported threshold mode: {threshold!r}")
    elif threshold is None:
        cutoff = float(np.median(ref_values))
    else:
        cutoff = float(threshold)

    ref_mask = ref_values >= cutoff
    target_mask = feature_matrix >= cutoff
    correlation = np.corrcoef(feature_matrix[0], ref_values)[0, 1]
    if not np.isfinite(correlation):
        correlation = 0.0
    intersection = np.logical_and(ref_mask, target_mask[0]).sum()
    ref_total = int(ref_mask.sum())
    target_total = int(target_mask[0].sum())
    min_total = min(ref_total, target_total)
    moc = 0.0 if min_total == 0 else intersection / min_total
    m1 = 0.0 if ref_total == 0 else intersection / ref_total
    m2 = 0.0 if target_total == 0 else intersection / target_total
    denominator = ref_total + target_total
    dice = 0.0 if denominator == 0 else 2 * intersection / denominator
    return {
        "cor": float(correlation),
        "MOC": float(moc),
        "M1": float(m1),
        "M2": float(m2),
        "Dice": float(dice),
    }


def colocalized(
    obj: SpectralImagingExperiment,
    i: Any = None,
    mz: Any = None,
    ref: Any = None,
    threshold: Any = "median",
    n: float = np.inf,
    sort_by: str = "cor",
) -> pd.DataFrame | list[pd.DataFrame]:
    """Rank features by colocalization against a reference image or feature.

    ``i`` selects a feature index, ``mz`` selects matching feature m/z values, and
    ``ref`` may be a numeric pixel vector or a logical mask. The returned DataFrame
    contains the ranking columns plus feature metadata.
    """
    if not isinstance(obj, SpectralImagingExperiment):
        raise TypeError("colocalized requires a shared-domain imaging experiment")
    if sort_by not in {"cor", "MOC", "M1", "M2", "Dice", "none"}:
        options = "{'cor', 'MOC', 'M1', 'M2', 'Dice', 'none'}"
        raise ValueError(f"sort_by must be one of {options}")

    experiment = cast(MSImagingExperiment, obj)
    references = _feature_reference(experiment, i=i, mz=mz, ref=ref)
    if references.ndim == 1:
        reference_vectors = [np.asarray(references, dtype=np.float64).ravel()]
    else:
        reference_vectors = [
            np.asarray(row, dtype=np.float64).ravel()
            for row in np.asarray(references, dtype=np.float64)
        ]

    results: list[pd.DataFrame] = []
    for reference in reference_vectors:
        scores = []
        for feature_index in range(experiment.shape[0]):
            feature_vector = np.asarray(
                experiment.intensity[feature_index], dtype=np.float64
            ).ravel()
            metric_values = _colocalization_scores(
                feature_vector.reshape(1, -1), reference, threshold
            )
            scores.append(
                {
                    "i": int(feature_index),
                    "mz": float(experiment.mz[feature_index]),
                    **metric_values,
                }
            )
        frame = pd.DataFrame(scores)
        if sort_by != "none":
            frame = frame.sort_values(by=sort_by, ascending=False, kind="mergesort")
        limit = len(frame) if np.isinf(n) else max(0, int(n))
        results.append(frame.head(limit).reset_index(drop=True))

    if len(results) == 1:
        return results[0]
    return results


def slice_image(
    obj: SpectralImagingExperiment,
    i: Any = None,
    run: Any = None,
    simplify: bool = True,
    drop: bool = True,
    **conditions: Any,
) -> Any:
    """Extract selected feature intensities as 2D ion-image rasters.

    Coordinates are rasterized with rows ordered by ascending ``y`` and columns
    by ascending ``x``. Missing grid positions are NaN. With ``simplify=True``,
    results are stacked as ``(feature, run, y, x)`` and singleton dimensions are
    removed when ``drop=True``. Otherwise a feature-major list of per-run image
    lists is returned. Three-dimensional coordinates are not yet rasterized.
    """
    if not isinstance(obj, SpectralImagingExperiment):
        raise TypeError("slice_image requires a shared-domain imaging experiment")
    if "z" in obj.coord.columns:
        raise NotImplementedError("slice_image currently supports 2D coordinates")
    selected = (
        features(obj, **conditions) if i is None else _selected_indices(i, obj.shape[0])
    )
    if i is not None and conditions:
        condition_indices = features(obj, **conditions)
        selected = selected[np.isin(selected, condition_indices)]

    all_runs = list(pd.unique(obj.pixel_data["run"]))
    if run is None:
        selected_runs = all_runs
    elif isinstance(run, (str, bytes)):
        selected_runs = [run]
    elif np.isscalar(run):
        run_index = int(np.asarray(run).item())
        if run_index < 0:
            run_index += len(all_runs)
        if run_index < 0 or run_index >= len(all_runs):
            raise IndexError("run index is out of range")
        selected_runs = [all_runs[run_index]]
    else:
        selected_runs = list(run)
    missing_runs = [value for value in selected_runs if value not in all_runs]
    if missing_runs:
        raise ValueError(f"unknown run labels: {missing_runs}")

    experiment = cast(MSImagingExperiment, obj)
    matrix = cast(Any, experiment.intensity)
    feature_images: list[list[NDArray[np.float64]]] = []
    for feature_index in selected:
        run_images = []
        for run_label in selected_runs:
            pixel_indices = np.flatnonzero(
                obj.pixel_data["run"].to_numpy() == run_label
            )
            coord = cast(Any, obj.coord).iloc[pixel_indices].reset_index(drop=True)
            if sparse.issparse(matrix):
                sparse_matrix = cast(Any, matrix)
                signal = np.asarray(
                    sparse_matrix[feature_index, pixel_indices].toarray()
                ).ravel()
            else:
                signal = np.asarray(matrix)[feature_index, pixel_indices]
            run_images.append(_rasterize(signal, coord))
        feature_images.append(run_images)
    if not simplify:
        return feature_images
    if not feature_images or not selected_runs:
        return np.empty((0, 0), dtype=float)
    stacked = np.stack([np.stack(row, axis=0) for row in feature_images], axis=0)
    if drop:
        stacked = np.squeeze(
            stacked,
            axis=tuple(
                axis for axis, size in enumerate(stacked.shape[:2]) if size == 1
            ),
        )
    return stacked
