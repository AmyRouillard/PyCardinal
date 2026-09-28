"""Deferred baseline estimation and subtraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter1d, median_filter, minimum_filter1d

from pycardinal.core import SpectralImagingData, add_processing


def _lower_hull(signal: np.ndarray) -> np.ndarray:
    points: list[int] = []
    indices = np.arange(len(signal))
    for index in indices:
        while len(points) >= 2:
            left, middle = points[-2:]
            cross = (middle - left) * (signal[index] - signal[left]) - (
                signal[middle] - signal[left]
            ) * (index - left)
            if cross > 0:
                break
            points.pop()
        points.append(int(index))
    return np.asarray(np.interp(indices, points, signal[points]), dtype=float)


def _snip_baseline(signal: np.ndarray, iterations: int) -> np.ndarray:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    transformed = np.log1p(np.log1p(np.sqrt(np.maximum(signal, 0) + 1)))
    clipped = transformed.copy()
    for width in range(min(iterations, max((len(signal) - 1) // 2, 0)), 0, -1):
        center = clipped[width:-width]
        neighbors = (clipped[: -2 * width] + clipped[2 * width :]) / 2
        clipped[width:-width] = np.minimum(center, neighbors)
    return np.asarray(np.square(np.expm1(np.expm1(clipped))) - 1, dtype=float)


def _apply_baseline(
    intensity: ArrayLike,
    mz: ArrayLike,
    *,
    method: str,
    window: int,
    iterations: int,
) -> np.ndarray:
    signal = np.asarray(intensity, dtype=float)
    if signal.ndim != 1:
        raise ValueError("baseline reduction expects a one-dimensional spectrum")
    if window < 1:
        raise ValueError("window must be positive")
    if method == "locmin":
        baseline = minimum_filter1d(signal, size=window, mode="nearest")
        baseline = gaussian_filter1d(
            baseline,
            sigma=max(window / 4, 0.5),
            mode="nearest",
        )
    elif method == "hull":
        baseline = _lower_hull(signal)
    elif method == "snip":
        baseline = _snip_baseline(signal, iterations)
    elif method == "median":
        baseline = median_filter(signal, size=window, mode="nearest")
    else:
        raise ValueError(f"unsupported baseline method: {method!r}")
    return np.asarray(np.maximum(signal - baseline, 0), dtype=float)


def reduce_baseline(
    obj: SpectralImagingData,
    method: str = "locmin",
    *,
    window: int = 31,
    iterations: int = 40,
) -> SpectralImagingData:
    """Queue baseline estimation and subtraction for every spectrum.

    Methods are ``locmin``, ``hull``, ``snip``, and ``median``. The returned
    dataset is a copy with a deferred step; call ``process`` to apply it.
    Output intensities are clipped at zero after baseline subtraction.
    """
    methods = {"locmin", "hull", "snip", "median"}
    if method not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}")
    return add_processing(
        obj,
        _apply_baseline,
        label="baseline reduction",
        metadata={"method": method},
        method=method,
        window=window,
        iterations=iterations,
    )
