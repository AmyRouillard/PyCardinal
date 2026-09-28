"""Deferred smoothing filters for mass spectra."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import savgol_filter

from pycardinal.core import SpectralImagingData, add_processing


def _bilateral(
    signal: np.ndarray,
    width: float,
    sigma_range: float | None,
    adaptive: bool,
) -> np.ndarray:
    radius = max(1, int(np.ceil(width)))
    sigma_space = max(width / 2, 0.5)
    global_scale = float(np.std(signal))
    base_range = sigma_range or max(global_scale * 0.25, np.finfo(float).eps)
    padded = np.pad(signal, radius, mode="edge")
    result = np.empty_like(signal)
    for index, center in enumerate(signal):
        window = padded[index : index + 2 * radius + 1]
        local_scale = float(np.std(window)) if adaptive else base_range
        intensity_scale = max(
            local_scale if adaptive else base_range,
            np.finfo(float).eps,
        )
        distance = np.arange(-radius, radius + 1, dtype=float)
        weights = np.exp(-0.5 * (distance / sigma_space) ** 2)
        weights *= np.exp(-0.5 * ((window - center) / intensity_scale) ** 2)
        total = float(weights.sum())
        result[index] = float(np.dot(weights, window) / total) if total else center
    return result


def _guided(
    signal: np.ndarray,
    width: int,
    epsilon: float,
    peak_aware: bool,
) -> np.ndarray:
    if len(signal) < 3:
        return signal.copy()
    window = max(3, width if width % 2 else width + 1)
    mean = uniform_filter1d(signal, size=window, mode="nearest")
    mean_square = uniform_filter1d(signal * signal, size=window, mode="nearest")
    variance = np.maximum(mean_square - mean * mean, 0)
    coefficient = variance / (variance + max(epsilon, np.finfo(float).eps))
    intercept = mean - coefficient * mean
    result = uniform_filter1d(
        coefficient, size=window, mode="nearest"
    ) * signal + uniform_filter1d(intercept, size=window, mode="nearest")
    if peak_aware:
        local_max = uniform_filter1d(signal, size=window, mode="nearest")
        result = np.maximum(result, np.minimum(signal, local_max + np.sqrt(variance)))
    return np.asarray(result, dtype=float)


def _diffusion(
    signal: np.ndarray,
    iterations: int,
    kappa: float,
    step: float,
) -> np.ndarray:
    result = signal.copy()
    scale = max(kappa, np.finfo(float).eps)
    for _ in range(iterations):
        difference = np.diff(result)
        conductance = np.exp(-np.square(difference / scale))
        flux = conductance * difference
        update = np.zeros_like(result)
        update[:-1] += flux
        update[1:] -= flux
        result += step * update
    return result


def _apply_smoothing(
    intensity: ArrayLike,
    mz: ArrayLike,
    *,
    method: str,
    width: float,
    sigma: float | None,
    sigma_range: float | None,
    polyorder: int,
    iterations: int,
    kappa: float | None,
    step: float,
    epsilon: float,
) -> np.ndarray:
    signal = np.asarray(intensity, dtype=float)
    if signal.ndim != 1:
        raise ValueError("smoothing expects a one-dimensional spectrum")
    if width <= 0 or not np.isfinite(width):
        raise ValueError("width must be a positive finite value")
    if method == "gaussian":
        return np.asarray(
            gaussian_filter1d(
                signal,
                sigma=sigma or max(width / 2, 0.5),
                mode="nearest",
            ),
            dtype=float,
        )
    if method == "ma":
        window = max(1, int(round(width)))
        return uniform_filter1d(signal, size=window, mode="nearest")
    if method == "sgolay":
        if len(signal) < 3:
            return signal.copy()
        window = min(
            len(signal) if len(signal) % 2 else len(signal) - 1,
            max(3, int(round(width)) | 1),
        )
        return np.asarray(
            savgol_filter(
                signal,
                window_length=window,
                polyorder=min(polyorder, window - 1),
                mode="interp",
            ),
            dtype=float,
        )
    if method == "bilateral":
        return _bilateral(signal, width, sigma_range, adaptive=False)
    if method == "adaptive":
        return _bilateral(signal, width, sigma_range, adaptive=True)
    if method == "diff":
        if iterations < 0 or not 0 < step <= 0.25:
            raise ValueError("diffusion requires iterations >= 0 and step in (0, 0.25]")
        return _diffusion(signal, iterations, kappa or float(np.std(signal)), step)
    if method == "guide":
        return _guided(signal, int(round(width)), epsilon, peak_aware=False)
    if method == "pag":
        return _guided(signal, int(round(width)), epsilon, peak_aware=True)
    raise ValueError(f"unsupported smoothing method: {method!r}")


def smooth(
    obj: SpectralImagingData,
    method: str = "gaussian",
    *,
    width: float = 5,
    sigma: float | None = None,
    sigma_range: float | None = None,
    polyorder: int = 2,
    iterations: int = 5,
    kappa: float | None = None,
    step: float = 0.2,
    epsilon: float = 1e-3,
) -> SpectralImagingData:
    """Queue a spectral smoothing method for later execution by ``process``.

    Supported methods are ``gaussian``, ``bilateral``, ``adaptive``, ``diff``,
    ``guide``, ``pag``, ``sgolay``, and ``ma``. ``width`` is in sample points.
    The advanced methods are pragmatic SciPy/NumPy implementations, not exact
    numeric reproductions of Cardinal's matter filters.
    """
    methods = {
        "gaussian",
        "bilateral",
        "adaptive",
        "diff",
        "guide",
        "pag",
        "sgolay",
        "ma",
    }
    if method not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}")
    options: dict[str, Any] = {
        "method": method,
        "width": width,
        "sigma": sigma,
        "sigma_range": sigma_range,
        "polyorder": polyorder,
        "iterations": iterations,
        "kappa": kappa,
        "step": step,
        "epsilon": epsilon,
    }
    return add_processing(
        obj,
        _apply_smoothing,
        label="smoothing",
        metadata={"method": method, "width": width},
        **options,
    )
