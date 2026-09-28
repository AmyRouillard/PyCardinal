"""ROI selection and categorical-mask helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.path import Path
from numpy.typing import NDArray


def make_factor(*, ordered: bool = False, **named_masks: Any) -> Any:
    """Combine named boolean masks into a first-match categorical factor."""
    if not named_masks:
        raise ValueError("at least one named mask is required")
    names = list(named_masks)
    masks = [np.asarray(named_masks[name], dtype=bool) for name in names]
    length = len(masks[0])
    if any(mask.ndim != 1 or len(mask) != length for mask in masks):
        raise ValueError("all masks must be one-dimensional and equally sized")
    values = np.full(length, -1, dtype=int)
    for index, mask in enumerate(masks):
        values[(values < 0) & mask] = index
    labels = np.asarray(names, dtype=object)
    result = np.empty(length, dtype=object)
    result[:] = None
    matched = values >= 0
    result[matched] = labels[values[matched]]
    return pd.Categorical(result, categories=names, ordered=ordered)


def _coordinates(obj: Any) -> np.ndarray:
    if not hasattr(obj, "coord"):
        raise TypeError("ROI selection requires an object with coordinates")
    return np.asarray(obj.coord.loc[:, ["x", "y"]], dtype=float)


def select_roi(
    obj: Any,
    mode: str = "region",
    polygon: Any = None,
    points: Any = None,
    tolerance: float = 0.5,
    **_: Any,
) -> NDArray[np.bool_]:
    """Return a pixel mask from a polygon or selected coordinate points.

    ``polygon`` and ``points`` provide a deterministic, noninteractive path for
    scripts and tests. When neither is provided, an interactive Matplotlib
    polygon selector is opened for ``mode="region"``.
    """
    if mode not in {"region", "pixels"}:
        raise ValueError("mode must be 'region' or 'pixels'")
    coordinates = _coordinates(obj)
    if mode == "region":
        if polygon is None:
            if points is None:
                raise ValueError(
                    "polygon is required for noninteractive region selection"
                )
            polygon = points
        vertices = np.asarray(polygon, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("polygon must contain at least three x/y vertices")
        return Path(vertices).contains_points(coordinates)
    if points is None:
        raise ValueError("points is required for pixel selection")
    selected_points = np.asarray(points, dtype=float)
    if selected_points.ndim != 2 or selected_points.shape[1] != 2:
        raise ValueError("points must be an N x 2 array")
    distances = np.linalg.norm(
        coordinates[:, None, :] - selected_points[None, :, :], axis=2
    )
    mask: NDArray[np.bool_] = np.asarray(
        np.any(distances <= float(tolerance), axis=1), dtype=np.bool_
    )
    return mask


__all__ = ["make_factor", "select_roi"]
