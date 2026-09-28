"""Spatial neighborhood, weighting, and distance utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import pairwise_distances  # type: ignore[import-untyped]

from pycardinal.core.imaging_data import (
    MSImagingExperiment,
    SpectralImagingData,
    SpectralImagingExperiment,
)
from pycardinal.core.metadata import PositionDataFrame

NeighborList = list[NDArray[np.intp]]


def _coordinates(value: Any) -> NDArray[np.float64]:
    if isinstance(value, SpectralImagingData):
        value = value.coord
    if isinstance(value, PositionDataFrame):
        value = value.coord
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    coordinates = np.asarray(value, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 1:
        raise ValueError("coordinates must be a two-dimensional array")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must contain only finite values")
    return coordinates


def _neighbor_list(neighbors: Sequence[ArrayLike]) -> NeighborList:
    result = [np.asarray(indices, dtype=np.intp) for indices in neighbors]
    if any(indices.ndim != 1 for indices in result):
        raise ValueError("each neighbor entry must be one-dimensional")
    return result


def _weight_list(weights: Sequence[ArrayLike]) -> list[NDArray[np.float64]]:
    result = [np.asarray(values, dtype=np.float64) for values in weights]
    if any(values.ndim != 1 or not np.isfinite(values).all() for values in result):
        raise ValueError("each neighborhood weight entry must be a finite vector")
    return result


def find_neighbors(
    coord_or_obj: Any,
    r: float = 1,
    groups: ArrayLike | None = None,
    metric: str = "maximum",
    p: float = 2,
    matrix: bool = False,
) -> NeighborList | sparse.csr_matrix:
    """Find each point's neighbors within radius ``r``.

    The point itself is included. ``metric="maximum"`` uses Chebyshev distance;
    ``euclidean`` and ``minkowski`` use the supplied Minkowski ``p``. Group labels
    prevent neighbors from crossing run or other group boundaries.
    """
    coordinates = _coordinates(coord_or_obj)
    if not np.isfinite(r) or r < 0:
        raise ValueError("r must be a finite nonnegative radius")
    if metric not in {"maximum", "chebyshev", "euclidean", "minkowski"}:
        raise ValueError("metric must be maximum, chebyshev, euclidean, or minkowski")
    minkowski_p = np.inf if metric in {"maximum", "chebyshev"} else p
    if minkowski_p < 1:
        raise ValueError("p must be at least 1")

    count = len(coordinates)
    labels = np.zeros(count, dtype=object) if groups is None else np.asarray(groups)
    if labels.ndim != 1 or len(labels) != count:
        raise ValueError("groups must contain one label per coordinate")

    candidates = cKDTree(coordinates).query_ball_point(
        coordinates, r=float(r), p=minkowski_p
    )
    result: list[NDArray[np.intp]] = []
    for center, candidate in enumerate(candidates):
        candidate_indices = np.asarray(candidate, dtype=np.intp)
        filtered = np.asarray(
            [
                int(index)
                for index in candidate_indices
                if labels[int(index)] == labels[int(center)]
            ],
            dtype=np.intp,
        )
        result.append(filtered)
    if matrix:
        row_indices = np.repeat(np.arange(count), [len(indices) for indices in result])
        column_indices = np.concatenate(result) if count else np.empty(0, dtype=np.intp)
        return sparse.csr_matrix(
            (np.ones(len(column_indices)), (row_indices, column_indices)),
            shape=(count, count),
        )
    return result


def _pixel_vectors(value: Any, byrow: bool) -> Any:
    if isinstance(value, SpectralImagingExperiment):
        matrix = cast(Any, value).intensity
        return matrix if byrow is False else matrix.T
    if sparse.issparse(value):
        sparse_value = cast(Any, value)
        return sparse_value if byrow else sparse_value.T
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("data must be a two-dimensional matrix")
    return matrix if byrow else matrix.T


def _weight_lists(
    coordinates: NDArray[np.float64],
    neighbors: NeighborList,
    sd: float | None,
    values: Any | None,
    adaptive: bool,
) -> list[NDArray[np.float64]]:
    default_sd = ((2.0 * 1.0) + 1.0) / 4.0 if sd is None else float(sd)
    if not np.isfinite(default_sd) or default_sd <= 0:
        raise ValueError("sd must be a finite positive value")
    output: list[NDArray[np.float64]] = []
    for center, indices in enumerate(neighbors):
        index_array = np.asarray(indices, dtype=np.intp)
        coordinate_distances = np.linalg.norm(
            coordinates[index_array] - coordinates[center], axis=1
        )
        weights = np.exp(-(coordinate_distances**2) / (2 * default_sd**2))
        if adaptive:
            if values is None:
                raise ValueError("adaptive weights require data values")
            matrix_values = np.asarray(values, dtype=np.float64)
            local = matrix_values[index_array]
            center_value = matrix_values[center : center + 1]
            distances = pairwise_distances(local, center_value).ravel()
            local_sd = max(float(distances.max(initial=0.0)) / 2.0, np.finfo(float).eps)
            weights *= np.exp(-(distances**2) / (2 * local_sd**2))
        output.append(np.asarray(weights, dtype=float))
    return output


def spatial_weights(
    x: Any,
    coord: ArrayLike | PositionDataFrame | None = None,
    r: float = 1,
    neighbors: Sequence[ArrayLike] | None = None,
    weights: str = "gaussian",
    sd: float | None = None,
    matrix: bool = False,
) -> list[NDArray[np.float64]] | sparse.csr_matrix:
    """Calculate Gaussian spatial weights, optionally modulated by data similarity.

    For an imaging experiment, pixel coordinates and feature vectors are inferred.
    Otherwise ``x`` supplies coordinates unless ``coord`` is explicitly provided.
    Adaptive weights multiply the spatial Gaussian by a per-neighborhood Gaussian
    based on Euclidean distances between data vectors.
    """
    if weights not in {"gaussian", "adaptive"}:
        raise ValueError("weights must be 'gaussian' or 'adaptive'")
    experiment = x if isinstance(x, SpectralImagingExperiment) else None
    if experiment is not None and coord is None:
        experiment_data = cast(MSImagingExperiment, experiment)
        coordinates = _coordinates(experiment_data.coord)
        data = _pixel_vectors(experiment_data.intensity, byrow=False)
    else:
        coordinates = _coordinates(x if coord is None else coord)
        data = None if coord is None else _pixel_vectors(x, byrow=True)
    if neighbors is None:
        candidate_neighbors = find_neighbors(coordinates, r=r)
        neighbor_indices = cast(NeighborList, candidate_neighbors)
    else:
        neighbor_indices = _neighbor_list(neighbors)
    if len(neighbor_indices) != len(coordinates):
        raise ValueError("neighbors must contain one list per coordinate")
    weight_lists = _weight_lists(
        coordinates,
        neighbor_indices,
        sd if sd is not None else (2 * r + 1) / 4,
        data,
        adaptive=weights == "adaptive",
    )
    if not matrix:
        return weight_lists
    row_indices = np.repeat(
        np.arange(len(coordinates)), [len(row) for row in weight_lists]
    )
    column_indices = (
        np.concatenate(neighbor_indices)
        if len(coordinates)
        else np.empty(0, dtype=np.intp)
    )
    values = np.concatenate(weight_lists) if len(coordinates) else np.empty(0)
    return sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(coordinates), len(coordinates)),
    )


def spatial_dists(
    x: Any,
    y: Any,
    coord: ArrayLike | PositionDataFrame | None = None,
    r: float = 1,
    neighbors: Sequence[ArrayLike] | None = None,
    neighbors_weights: Sequence[ArrayLike] | None = None,
    weights: ArrayLike | None = None,
    byrow: bool = True,
    metric: str = "euclidean",
    p: float = 2,
) -> NDArray[np.float64]:
    """Return neighborhood-weighted distances from observations in ``x`` to ``y``.

    The result has one row per center coordinate and one column per observation
    in ``y``. Set ``byrow=False`` for a feature-by-pixel matrix.
    """
    experiment = x if isinstance(x, SpectralImagingExperiment) else None
    if experiment is not None:
        experiment_data = cast(MSImagingExperiment, experiment)
        coordinates = _coordinates(experiment_data.coord if coord is None else coord)
        left = _pixel_vectors(experiment_data.intensity, byrow=False)
    else:
        if coord is None:
            raise ValueError("coord is required when x is not an imaging experiment")
        coordinates = _coordinates(coord)
        left = _pixel_vectors(x, byrow=byrow)
    right = _pixel_vectors(y, byrow=True)
    if left.shape[0] != len(coordinates):
        raise ValueError("x must contain one observation per coordinate")
    if neighbors is None:
        found = find_neighbors(coordinates, r=r)
        neighbor_indices = cast(NeighborList, found)
    else:
        neighbor_indices = _neighbor_list(neighbors)
    if len(neighbor_indices) != len(coordinates):
        raise ValueError("neighbors must contain one list per coordinate")
    if neighbors_weights is not None and weights is not None:
        raise ValueError("provide either neighbors_weights or weights, not both")
    weight_source = neighbors_weights if neighbors_weights is not None else weights
    if weight_source is None:
        weight_lists = cast(
            list[NDArray[np.float64]],
            spatial_weights(coordinates, r=r, neighbors=neighbor_indices),
        )
    else:
        weight_lists = _weight_list(
            cast(Sequence[ArrayLike], [weight_source])
            if isinstance(weight_source, np.ndarray) and weight_source.ndim == 1
            else cast(Sequence[ArrayLike], weight_source)
        )
    if len(weight_lists) != len(coordinates):
        raise ValueError("weights must contain one list per coordinate")
    distances = pairwise_distances(
        left, right, metric=metric, **({"p": p} if metric == "minkowski" else {})
    )
    result = np.empty((len(coordinates), distances.shape[1]), dtype=float)
    for center, (indices, local_weights) in enumerate(
        zip(neighbor_indices, weight_lists)
    ):
        if len(indices) != len(local_weights) or np.any(local_weights < 0):
            raise ValueError("each neighborhood needs matching nonnegative weights")
        total = local_weights.sum()
        if total <= 0:
            raise ValueError("neighborhood weights must sum to a positive value")
        result[center] = np.average(distances[indices], axis=0, weights=local_weights)
    return result
