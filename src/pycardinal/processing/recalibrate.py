"""Deferred m/z-axis recalibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

from pycardinal.core import SpectralImagingData, add_processing
from pycardinal.core.metadata import MassDataFrame


def _apply_recalibration(
    intensity: ArrayLike,
    mz: ArrayLike,
    *,
    ref: ArrayLike,
    method: str,
    tolerance: float | None,
    units: str,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(intensity, dtype=float)
    axis = np.asarray(mz, dtype=float)
    reference = np.asarray(ref, dtype=float)
    if signal.size < 3 or reference.size == 0:
        return axis.copy(), signal.copy()
    peak_indices, _ = find_peaks(signal)
    if not peak_indices.size:
        return axis.copy(), signal.copy()
    observed = axis[peak_indices]
    nearest = np.searchsorted(reference, observed).clip(0, len(reference) - 1)
    left = np.maximum(nearest - 1, 0)
    use_left = np.abs(observed - reference[left]) < np.abs(
        observed - reference[nearest]
    )
    nearest = np.where(use_left, left, nearest)
    error = np.abs(observed - reference[nearest])
    if units == "ppm":
        error = error / reference[nearest] * 1e6
    allowed = np.inf if tolerance is None else tolerance
    matched = error <= allowed
    observed = observed[matched]
    targets = reference[nearest[matched]]
    if not observed.size:
        return axis.copy(), signal.copy()
    order = np.argsort(observed)
    observed = observed[order]
    targets = targets[order]
    unique = np.concatenate(([True], np.diff(observed) > 0))
    observed = observed[unique]
    targets = targets[unique]
    if len(observed) == 1:
        corrected = axis + (targets[0] - observed[0])
    elif method == "dtw" and len(observed) >= 3:
        corrected = np.asarray(
            PchipInterpolator(observed, targets, extrapolate=True)(axis),
            dtype=float,
        )
    else:
        corrected = np.interp(axis, observed, targets)
        left_shift = targets[0] - observed[0]
        right_shift = targets[-1] - observed[-1]
        corrected[axis < observed[0]] = axis[axis < observed[0]] + left_shift
        corrected[axis > observed[-1]] = axis[axis > observed[-1]] + right_shift
    if np.any(np.diff(corrected) <= 0):
        raise ValueError(
            "reference matches produce a non-monotonic calibrated m/z axis"
        )
    return corrected, signal.copy()


def recalibrate(
    obj: SpectralImagingData,
    ref: ArrayLike | MassDataFrame,
    method: str = "locmax",
    tolerance: float | None = None,
    units: str = "ppm",
) -> SpectralImagingData:
    """Queue mass-axis calibration toward reference peak locations.

    ``locmax``, ``dtw``, and ``cow`` are supported. The current Python port
    matches local maxima to the nearest reference and interpolates a monotone
    warp; ``dtw`` uses shape-preserving cubic interpolation when enough anchors
    are present. This is a practical approximation, not a bitwise port of the
    three Cardinal/matter algorithms.
    """
    if method not in {"locmax", "dtw", "cow"}:
        raise ValueError("method must be 'locmax', 'dtw', or 'cow'")
    if units not in {"ppm", "mz"}:
        raise ValueError("units must be 'ppm' or 'mz'")
    reference = (
        ref.mz if isinstance(ref, MassDataFrame) else np.asarray(ref, dtype=float)
    )
    if (
        reference.ndim != 1
        or not np.isfinite(reference).all()
        or np.any(np.diff(reference) <= 0)
    ):
        raise ValueError("ref must be a finite, strictly increasing m/z vector")
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be finite and nonnegative")
    return add_processing(
        obj,
        _apply_recalibration,
        label="m/z recalibration",
        metadata={"method": method, "tolerance": tolerance, "units": units},
        ref=reference,
        method=method,
        tolerance=tolerance,
        units=units,
    )
