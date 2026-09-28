"""Matplotlib plotting helpers for spectral imaging data."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from pycardinal.core.imaging_data import MSImagingArrays, MSImagingExperiment
from pycardinal.features import slice_image


def _indices(value: Any, length: int) -> list[int]:
    if value is None:
        return list(range(length))
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(index) for index in np.asarray(value).ravel()]


def plot_spectra(
    obj: MSImagingExperiment | MSImagingArrays,
    i: ArrayLike | int | None = None,
    superpose: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot selected spectra and return the Matplotlib axes."""
    axis = plt.gca() if ax is None else ax
    if isinstance(obj, MSImagingExperiment):
        indices = _indices(i, obj.shape[1])
        for index in indices:
            axis.plot(obj.mz, np.asarray(obj.intensity)[:, index], **kwargs)
    elif isinstance(obj, MSImagingArrays):
        indices = _indices(i, len(obj))
        for index in indices:
            axis.plot(obj.mz[index], obj.intensity[index], **kwargs)
    else:
        raise TypeError("plot_spectra requires a mass-spectrometry imaging object")
    axis.set_xlabel("m/z")
    axis.set_ylabel("Intensity")
    if xlim is not None:
        axis.set_xlim(xlim)
    if ylim is not None:
        axis.set_ylim(ylim)
    if (
        superpose is False
        and len(axis.lines) > 1
        and any(not str(line.get_label()).startswith("_") for line in axis.lines)
    ):
        axis.legend()
    return axis


def _image_data(obj: Any, feature: Any, i: Any) -> tuple[np.ndarray, list[str]]:
    if hasattr(obj, "intensity") and isinstance(obj, MSImagingExperiment):
        if feature is not None:
            i = np.array([int(np.argmin(np.abs(obj.mz - float(feature))))])
        if i is None:
            i = [0]
        selected = _indices(i, obj.shape[0])
        labels = [f"m/z={obj.mz[index]:.4g}" for index in selected]
        return np.asarray(slice_image(obj, i=selected, drop=False)), labels
    raise TypeError("plot_image currently requires an MSImagingExperiment")


def plot_image(
    obj: Any,
    feature: float | None = None,
    i: int | ArrayLike | None = None,
    superpose: bool = False,
    scale: bool = False,
    ax: Any = None,
    cmap: str = "viridis",
    **kwargs: Any,
) -> Any:
    """Plot one or more ion images and return the Matplotlib axes."""
    images, labels = _image_data(obj, feature, i)
    axis = plt.gca() if ax is None else ax
    if images.ndim == 4:
        images = images[:, 0]
    if images.ndim == 2:
        images = images[None, ...]
    for index, image in enumerate(images):
        values = image.astype(float, copy=True)
        if scale:
            maximum = np.nanmax(np.abs(values))
            if maximum > 0:
                values /= maximum
        if superpose and index > 0:
            axis.imshow(values, cmap=cmap, alpha=0.45, **kwargs)
        else:
            axis.imshow(values, cmap=cmap, **kwargs)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    if len(labels) > 1 and not superpose:
        axis.set_title(", ".join(labels))
    return axis


def plot_model(fit: Any, type: str = "scores", ax: Any = None, **kwargs: Any) -> Any:
    """Plot common result-object fields such as scores, centers, or clusters."""
    axis = plt.gca() if ax is None else ax
    if type in {"scores", "x"} and hasattr(fit, "scores"):
        values = np.asarray(fit.scores)
        second = values[:, 1] if values.shape[1] > 1 else np.zeros(len(values))
        axis.plot(values[:, 0], second, **kwargs)
    elif type in {"centers", "rotation", "loadings"}:
        values = np.asarray(getattr(fit, "centers", getattr(fit, "loadings")))
        axis.plot(values, **kwargs)
    elif type in {"cluster", "class"} and hasattr(fit, "cluster"):
        axis.scatter(np.arange(len(fit.cluster)), fit.cluster, **kwargs)
    elif type == "scree" and hasattr(fit, "sdev"):
        axis.plot(np.arange(1, len(fit.sdev) + 1), fit.sdev, marker="o", **kwargs)
    else:
        raise ValueError(f"unsupported plot type {type!r} for {fit.__class__.__name__}")
    return axis


def image_model(fit: Any, type: str = "x", ax: Any = None, **kwargs: Any) -> Any:
    """Plot a model's spatial values when pixel coordinates are attached."""
    axis = plt.gca() if ax is None else ax
    values = getattr(fit, "cluster", getattr(fit, "classes", None))
    if values is None:
        raise ValueError("fit does not contain spatial class values")
    axis.imshow(np.asarray(values).reshape(1, -1), aspect="auto", **kwargs)
    axis.set_title(type)
    return axis


__all__ = ["image_model", "plot_image", "plot_model", "plot_spectra"]
