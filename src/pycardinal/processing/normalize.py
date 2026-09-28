"""Deferred intensity normalization for spectral imaging data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from pycardinal.core import (
    MSImagingArrays,
    MSImagingExperiment,
    SpectralImagingData,
    add_processing,
)
from pycardinal.core.metadata import MassDataFrame


def _normalization_scale(obj: SpectralImagingData) -> int:
    if isinstance(obj, MSImagingExperiment):
        return max(obj.shape[0], 1)
    if isinstance(obj, MSImagingArrays):
        return max((len(spectrum) for spectrum in obj.mz), default=1)
    return 1


def _apply_normalization(
    intensity: ArrayLike,
    mz: ArrayLike,
    *,
    method: str,
    scale: float,
    ref: ArrayLike | None,
    tolerance: float | None,
    units: str,
) -> np.ndarray:
    signal = np.asarray(intensity, dtype=float)
    axis = np.asarray(mz, dtype=float)
    if method == "tic":
        denominator = float(np.sum(signal))
    elif method == "rms":
        denominator = float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0
    else:
        if ref is None:
            raise ValueError("ref is required for reference normalization")
        reference = np.asarray(ref, dtype=float)
        if reference.ndim == 0:
            reference = reference.reshape(1)
        if reference.ndim != 1 or not np.isfinite(reference).all():
            raise ValueError("ref must be a finite m/z vector")
        if units not in {"mz", "ppm"}:
            raise ValueError("units must be 'mz' or 'ppm'")
        indices = np.searchsorted(axis, reference).clip(0, max(len(axis) - 1, 0))
        if not len(axis):
            return signal.copy()
        left = np.maximum(indices - 1, 0)
        use_left = np.abs(axis[left] - reference) < np.abs(axis[indices] - reference)
        indices = np.where(use_left, left, indices)
        errors = np.abs(axis[indices] - reference)
        if units == "ppm":
            errors = errors / reference * 1e6
        accepted = errors <= (np.inf if tolerance is None else tolerance)
        denominator = (
            float(np.sum(signal[indices[accepted]])) if np.any(accepted) else 0.0
        )
    if denominator <= 0 or not np.isfinite(denominator):
        return signal.copy()
    return signal * (scale / denominator)


def normalize(
    obj: SpectralImagingData,
    method: str = "tic",
    scale: float | None = None,
    ref: ArrayLike | MassDataFrame | None = None,
    *,
    tolerance: float | None = None,
    units: str = "ppm",
) -> SpectralImagingData:
    """Queue TIC, RMS, or reference intensity normalization.

    Parameters
    ----------
    obj
        Mass imaging data with profile or centroided spectra.
    method
        ``"tic"`` scales total ion current, ``"rms"`` scales root-mean-square
        intensity, and ``"reference"`` scales the sum of peaks matching ``ref``.
    scale
        Target denominator after normalization. Defaults to the feature count
        (shared-axis data) or longest spectrum (ragged data); reference mode
        defaults to 1.
    ref
        Reference m/z values for ``method="reference"``.
    tolerance
        Maximum matching distance for reference peaks. Defaults to unlimited.
    units
        Tolerance units: ``"ppm"`` or absolute ``"mz"``.

    Returns
    -------
    SpectralImagingData
        A shallow dataset copy with normalization queued. Call ``process`` to
        apply the operation.
    """
    if method not in {"tic", "rms", "reference"}:
        raise ValueError("method must be 'tic', 'rms', or 'reference'")
    if method == "reference" and ref is None:
        raise ValueError("ref is required for reference normalization")
    if scale is None:
        scale = 1.0 if method == "reference" else float(_normalization_scale(obj))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite value")
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be finite and nonnegative")
    if units not in {"ppm", "mz"}:
        raise ValueError("units must be 'ppm' or 'mz'")
    if isinstance(ref, MassDataFrame):
        ref = ref.mz
    if isinstance(ref, Sequence) and not isinstance(ref, (str, bytes)):
        ref = np.asarray(ref, dtype=float)
    return add_processing(
        obj,
        _apply_normalization,
        label="intensity normalization",
        metadata={"method": method, "scale": scale},
        method=method,
        scale=float(scale),
        ref=ref,
        tolerance=tolerance,
        units=units,
    )
