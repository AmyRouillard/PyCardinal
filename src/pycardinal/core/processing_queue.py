"""Deferred per-spectrum processing for imaging datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from numpy.typing import ArrayLike
from scipy import sparse

from pycardinal.core.imaging_data import (
    MSImagingArrays,
    MSImagingExperiment,
    SpectralImagingData,
)
from pycardinal.core.metadata import MassDataFrame


@dataclass(frozen=True)
class ProcessingStep:
    """A queued spectrum function and the arguments needed to call it."""

    function: Callable[..., Any]
    label: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def add_processing(
    obj: SpectralImagingData,
    fn: Callable[..., Any],
    label: str,
    metadata: Mapping[str, Any] | None = None,
    **fn_kwargs: Any,
) -> SpectralImagingData:
    """Return a shallow copy with a processing function appended to its queue.

    The function is called as ``fn(intensity, mz, **fn_kwargs)``. It may return
    a new intensity vector or a pair ``(new_mz, new_intensity)``.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if not label:
        raise ValueError("label must be a non-empty string")
    result = obj.copy(deep=False)
    result.processing.append(
        ProcessingStep(fn, label, dict(metadata or {}), dict(fn_kwargs))
    )
    return result


def _apply_steps(
    mz: ArrayLike,
    intensity: ArrayLike,
    steps: list[ProcessingStep],
) -> tuple[np.ndarray, np.ndarray]:
    mass = np.asarray(mz).copy()
    signal = np.asarray(intensity).copy()
    for step in steps:
        output = step.function(signal, mass, **step.kwargs)
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError(
                    f"processing step {step.label!r} must return a two-item pair"
                )
            mass = np.asarray(output[0])
            signal = np.asarray(output[1])
        else:
            signal = np.asarray(output)
        if mass.ndim != 1 or signal.ndim != 1:
            raise ValueError(f"processing step {step.label!r} must return 1D arrays")
        if len(mass) != len(signal):
            raise ValueError(
                f"processing step {step.label!r} returned m/z and intensity lengths "
                "that differ"
            )
    return mass, signal


def _run_tasks(
    tasks: Sequence[tuple[ArrayLike, ArrayLike]],
    steps: list[ProcessingStep],
    n_jobs: int | None,
    chunk_size: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    size = chunk_size or max(len(tasks), 1)
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for offset in range(0, len(tasks), size):
        chunk = tasks[offset : offset + size]
        if n_jobs is None or n_jobs == 1:
            results.extend(_apply_steps(mz, signal, steps) for mz, signal in chunk)
        else:
            results.extend(
                Parallel(n_jobs=n_jobs)(
                    delayed(_apply_steps)(mz, signal, steps) for mz, signal in chunk
                )
            )
    return results


def process(
    obj: SpectralImagingData,
    *,
    n_jobs: int | None = None,
    chunk_size: int | None = None,
    verbose: bool = False,
) -> SpectralImagingData:
    """Apply queued processing steps and return a new dataset.

    Each step is applied to every spectrum in queue order. Shared-domain data
    must retain the same m/z axis across all pixels. File output is deferred to
    the I/O phase and is not accepted here.
    """
    del verbose
    steps = list(obj.processing)
    result = obj.copy(deep=False)
    result.processing = []
    if not steps:
        return result
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero")

    if isinstance(obj, MSImagingExperiment):
        matrix = obj.intensity
        tasks = [
            (
                obj.mz,
                (
                    np.asarray(cast(Any, matrix).getcol(index).toarray()).ravel()
                    if sparse.issparse(matrix)
                    else np.asarray(matrix)[:, index]
                ),
            )
            for index in range(obj.shape[1])
        ]
        processed = _run_tasks(tasks, steps, n_jobs, chunk_size)
        mass_axis = processed[0][0] if processed else obj.mz
        if any(not np.array_equal(mass_axis, mass) for mass, _ in processed[1:]):
            raise ValueError("shared-domain processing must produce the same m/z axis")
        result._spectra_data["intensity"] = (
            np.column_stack([signal for _, signal in processed])
            if processed
            else np.empty((obj.shape[0], 0))
        )
        if not np.array_equal(mass_axis, obj.mz):
            if len(mass_axis) == len(obj.mz):
                feature_data = obj.feature_data.copy()
                feature_data["mz"] = mass_axis
                result._feature_data = MassDataFrame(feature_data)
            else:
                result._feature_data = MassDataFrame(mz=mass_axis)
        return result

    if isinstance(obj, MSImagingArrays):
        tasks = list(zip(obj.mz, obj.intensity))
        processed = _run_tasks(tasks, steps, n_jobs, chunk_size)
        result._spectra_data["mz"] = [mass for mass, _ in processed]
        result._spectra_data["intensity"] = [signal for _, signal in processed]
        return result

    raise TypeError(f"unsupported dataset type: {type(obj).__name__}")


def reset(obj: SpectralImagingData) -> SpectralImagingData:
    """Return a shallow copy with all pending processing steps removed."""
    result = obj.copy(deep=False)
    result.processing = []
    return result
