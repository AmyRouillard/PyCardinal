"""Synthetic mass spectra and imaging experiments for examples and tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from pycardinal.core import (
    MassDataFrame,
    MSImagingArrays,
    MSImagingExperiment,
    PositionDataFrame,
)


def _rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _vector(
    values: ArrayLike,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional vector")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} values must be positive")
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{name} values must be nonnegative")
    return result


def _mass_domain(
    low: float,
    high: float,
    step: float,
    units: str,
    *,
    max_points: int = 2_000_000,
) -> NDArray[np.float64]:
    if not np.isfinite([low, high, step]).all() or low <= 0 or high < low or step <= 0:
        raise ValueError("from_, to, and by must define a positive increasing range")
    if units == "mz":
        count = int(np.floor((high - low) / step)) + 1
        if count > max_points:
            raise ValueError("requested spectrum domain exceeds 2 million points")
        return low + step * np.arange(count, dtype=float)
    if units != "ppm":
        raise ValueError("units must be 'ppm' or 'mz'")
    ratio = 1.0 + step * 1e-6
    count = int(np.floor(np.log(high / low) / np.log(ratio))) + 1
    if count > max_points:
        raise ValueError("requested spectrum domain exceeds 2 million points")
    return cast(
        NDArray[np.float64], low * np.power(ratio, np.arange(count, dtype=float))
    )


def _simulate_profiles(
    peak_mz: NDArray[np.float64],
    mean_intensity: NDArray[np.float64],
    domain: NDArray[np.float64],
    *,
    sdpeaks: ArrayLike | None,
    sdpeakmult: float,
    sdnoise: float,
    sdmz: float,
    resolution: float,
    fmax: float,
    baseline: float,
    decay: float,
    units: str,
    centroided: bool,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    n_peaks, n_spectra = mean_intensity.shape
    if sdpeaks is None:
        peak_sigma = sdpeakmult * np.log1p(np.maximum(mean_intensity, 0))
    else:
        supplied = np.asarray(sdpeaks, dtype=float)
        if supplied.ndim == 0:
            peak_sigma = np.full((n_peaks, n_spectra), float(supplied))
        elif supplied.ndim == 1 and supplied.size == n_peaks:
            peak_sigma = np.broadcast_to(supplied[:, None], (n_peaks, n_spectra))
        elif supplied.shape == (n_peaks, n_spectra):
            peak_sigma = supplied
        else:
            raise ValueError(
                "sdpeaks must be scalar, one value per peak, or peak-by-spectrum"
            )
        if not np.isfinite(peak_sigma).all() or np.any(peak_sigma < 0):
            raise ValueError("sdpeaks must be finite and nonnegative")

    heights = np.maximum(mean_intensity, 0) * np.exp(
        rng.normal(size=(n_peaks, n_spectra)) * peak_sigma
    )
    mz_error = rng.normal(size=(n_peaks, n_spectra)) * sdmz
    centers = (
        peak_mz[:, None] * (1 + mz_error * 1e-6)
        if units == "ppm"
        else peak_mz[:, None] + mz_error
    )
    if centroided:
        return heights

    if resolution <= 0 or not np.isfinite(resolution):
        raise ValueError("resolution must be a positive finite value")
    if not 0 < fmax < 1:
        raise ValueError("fmax must be between zero and one")
    if sdnoise < 0 or not np.isfinite(sdnoise):
        raise ValueError("sdnoise must be finite and nonnegative")
    if baseline < 0 or decay < 0:
        raise ValueError("baseline and decay must be nonnegative")

    profile = np.zeros((len(domain), n_spectra), dtype=float)
    for peak_index in range(n_peaks):
        width = max(float(peak_mz[peak_index] / resolution), np.finfo(float).eps)
        sigma = width / (2 * np.sqrt(2 * np.log(1 / fmax)))
        distance = (domain[:, None] - centers[peak_index][None, :]) / sigma
        profile += heights[peak_index][None, :] * np.exp(-0.5 * distance**2)
    if baseline:
        normalized = (domain - domain[0]) / max(domain[-1] - domain[0], 1.0)
        profile += baseline * np.exp(-decay * normalized[:, None])
    if sdnoise:
        noise = rng.normal(size=profile.shape) * sdnoise
        profile *= np.exp(noise - 0.5 * sdnoise**2)
    return np.maximum(profile, 0)


def simulate_spectra(
    n: int = 1,
    npeaks: int = 50,
    mz: ArrayLike | None = None,
    intensity: ArrayLike | None = None,
    from_: float | None = None,
    to: float | None = None,
    by: float = 400,
    sdpeaks: ArrayLike | None = None,
    sdpeakmult: float = 0.2,
    sdnoise: float = 0.1,
    sdmz: float = 10,
    resolution: float = 1000,
    fmax: float = 0.5,
    baseline: float = 0,
    decay: float = 10,
    units: str = "ppm",
    centroided: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> MassDataFrame:
    """Generate one or more reproducible profile or centroided spectra.

    Returns a ``MassDataFrame`` with one m/z value per row. A single spectrum
    uses an ``intensity`` column; multiple spectra use ``intensity_1`` through
    ``intensity_n``. Profile output is sampled on a shared domain. Centroided
    output uses the supplied/theoretical peak positions as its shared domain.
    """
    if n < 1 or npeaks < 1:
        raise ValueError("n and npeaks must be positive")
    if units not in {"ppm", "mz"}:
        raise ValueError("units must be 'ppm' or 'mz'")
    if not np.isfinite([sdpeakmult, sdmz]).all() or sdpeakmult < 0 or sdmz < 0:
        raise ValueError("sdpeakmult and sdmz must be nonnegative")
    if sdnoise < 0 or not np.isfinite(sdnoise):
        raise ValueError("sdnoise must be finite and nonnegative")
    if baseline < 0 or decay < 0:
        raise ValueError("baseline and decay must be nonnegative")
    if resolution <= 0 or not np.isfinite(resolution):
        raise ValueError("resolution must be a positive finite value")
    if not 0 < fmax < 1:
        raise ValueError("fmax must be between zero and one")
    rng = _rng(random_state)
    peak_mz = (
        np.sort(rng.lognormal(mean=7, sigma=0.3, size=npeaks))
        if mz is None
        else _vector(mz, "mz", positive=True)
    )
    peak_intensity = (
        rng.lognormal(mean=1, sigma=0.9, size=len(peak_mz))
        if intensity is None
        else _vector(intensity, "intensity", nonnegative=True)
    )
    if len(peak_mz) != len(peak_intensity):
        raise ValueError("mz and intensity must have the same length")
    supplied_sdpeaks = None if sdpeaks is None else np.asarray(sdpeaks, dtype=float)
    order = np.argsort(peak_mz)
    peak_mz = peak_mz[order]
    peak_intensity = peak_intensity[order]
    if supplied_sdpeaks is not None and supplied_sdpeaks.ndim == 1:
        if supplied_sdpeaks.size != len(order):
            raise ValueError("one-dimensional sdpeaks must have one value per peak")
        supplied_sdpeaks = supplied_sdpeaks[order]
    elif supplied_sdpeaks is not None and supplied_sdpeaks.ndim == 2:
        if supplied_sdpeaks.shape[0] != len(order):
            raise ValueError("two-dimensional sdpeaks must have one row per peak")
        supplied_sdpeaks = supplied_sdpeaks[order, :]
    sdpeaks = supplied_sdpeaks
    lower = 0.9 * float(peak_mz.min()) if from_ is None else float(from_)
    upper = 1.1 * float(peak_mz.max()) if to is None else float(to)
    domain = peak_mz if centroided else _mass_domain(lower, upper, by, units)
    intensity_matrix = np.broadcast_to(peak_intensity[:, None], (len(peak_mz), n))
    spectra = _simulate_profiles(
        peak_mz,
        intensity_matrix,
        domain,
        sdpeaks=sdpeaks,
        sdpeakmult=sdpeakmult,
        sdnoise=sdnoise,
        sdmz=sdmz,
        resolution=resolution,
        fmax=fmax,
        baseline=baseline,
        decay=decay,
        units=units,
        centroided=centroided,
        rng=rng,
    )
    data: dict[str, ArrayLike] = {"mz": domain}
    for spectrum_index in range(n):
        column_name = "intensity" if n == 1 else f"intensity_{spectrum_index + 1}"
        data[column_name] = spectra[:, spectrum_index]
    result = MassDataFrame(data)
    result.attrs["centroided"] = centroided
    result.attrs["random_state"] = (
        random_state if isinstance(random_state, int) else None
    )
    return result


def add_shape(
    pixel_data: PositionDataFrame | pd.DataFrame,
    center: Sequence[float] | Mapping[str, float],
    size: float,
    shape: str = "circle",
    name: str | None = None,
) -> PositionDataFrame:
    """Return pixel metadata with a circle or square ROI mask column added."""
    result = PositionDataFrame(pixel_data)
    coordinate_names = [column for column in ("x", "y", "z") if column in result]
    coordinates = result[coordinate_names].to_numpy(dtype=float)
    if isinstance(center, Mapping):
        try:
            center_values = np.asarray(
                [center[key] for key in coordinate_names], dtype=float
            )
        except KeyError as exc:
            raise ValueError(f"center must define coordinate {exc.args[0]!r}") from exc
    else:
        center_values = np.asarray(center, dtype=float)
        if center_values.ndim == 0:
            center_values = np.repeat(center_values, len(coordinate_names))
    if (
        center_values.shape != (len(coordinate_names),)
        or not np.isfinite(center_values).all()
    ):
        raise ValueError("center must have one finite value per coordinate dimension")
    if size < 0 or not np.isfinite(size):
        raise ValueError("size must be finite and nonnegative")
    if shape not in {"circle", "square"}:
        raise ValueError("shape must be 'circle' or 'square'")
    distance = np.abs(coordinates - center_values)
    mask = (
        np.linalg.norm(distance, axis=1) <= size
        if shape == "circle"
        else np.all(distance <= size, axis=1)
    )
    result[name or shape] = mask
    return PositionDataFrame(result)


def _lattice(dim: Sequence[int]) -> PositionDataFrame:
    dimensions = tuple(int(value) for value in dim)
    if len(dimensions) not in (2, 3) or any(value < 1 for value in dimensions):
        raise ValueError("dim must contain two or three positive integers")
    if len(dimensions) == 2:
        nx, ny = dimensions
        frame = pd.DataFrame(
            [(x, y) for y in range(1, ny + 1) for x in range(1, nx + 1)],
            columns=["x", "y"],
        )
        frame["run"] = "run0"
        return PositionDataFrame(frame)
    nx, ny, nz = dimensions
    frame = pd.DataFrame(
        [
            (x, y, z)
            for z in range(1, nz + 1)
            for y in range(1, ny + 1)
            for x in range(1, nx + 1)
        ],
        columns=["x", "y", "z"],
    )
    frame["run"] = [f"run{z - 1}" for z in frame["z"]]
    return PositionDataFrame(frame)


def _add_regions(
    pixels: PositionDataFrame,
    regions: Mapping[str, tuple[Sequence[float], float, str]],
) -> PositionDataFrame:
    result = pixels
    for name, (center, size, shape) in regions.items():
        result = add_shape(result, center, size, shape=shape, name=name)
    return result


def _positive_heights(
    rng: np.random.Generator,
    indices: NDArray[np.int64],
    mean: float,
    sd: float,
    npeaks: int,
) -> NDArray[np.float64]:
    result = np.zeros(npeaks, dtype=float)
    if indices.size:
        result[indices] = np.maximum(0.0, rng.normal(mean, sd, size=indices.size))
    return result


def preset_image_def(
    preset: int = 1,
    nrun: int = 1,
    npeaks: int = 30,
    dim: Sequence[int] = (20, 20),
    peakheight: float | Sequence[float] = np.e,
    peakdiff: float | Sequence[float] = np.e,
    sdsample: float = 0.2,
    jitter: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, PositionDataFrame | MassDataFrame]:
    """Build pixel/feature designs for one of nine example image presets.

    Presets provide compact, deterministic-in-design examples rather than
    reproductions of Cardinal's R random-number stream. Presets 1-8 are 2D;
    preset 9 creates a 3D lattice when ``dim`` has three entries.
    """
    if preset == 0:
        raise ValueError("preset must be a nonzero integer from 1 to 9")
    preset_number = (abs(int(preset)) - 1) % 9 + 1
    if nrun < 1 or npeaks < 1:
        raise ValueError("nrun and npeaks must be positive")
    if sdsample < 0 or not np.isfinite(sdsample):
        raise ValueError("sdsample must be finite and nonnegative")
    rng = _rng(random_state)
    dimensions = tuple(int(value) for value in dim)
    if preset_number == 9 and len(dimensions) == 2:
        dimensions = (*dimensions, nrun)
    if len(dimensions) != 2 and preset_number != 9:
        raise ValueError("only preset 9 supports three-dimensional designs")
    base = _lattice(dimensions)
    nx = dimensions[0]
    ny = dimensions[1]
    jitter_scale = 0.05 if jitter else 0.0
    x_jitter = rng.normal(0, nx * jitter_scale, size=len(base))
    y_jitter = rng.normal(0, ny * jitter_scale, size=len(base))
    base["x"] = base["x"].to_numpy(dtype=float) + x_jitter
    base["y"] = base["y"].to_numpy(dtype=float) + y_jitter
    center = {"x": (nx + 1) / 2, "y": (ny + 1) / 2}
    radius = max(1.0, min(nx, ny) / 3)
    if preset_number == 9:
        center["z"] = (dimensions[2] + 1) / 2
        radius = max(1.0, min(dimensions) / 3)

    if preset_number in {4, 5}:
        frames = []
        for condition in ("A", "B"):
            for run_index in range(nrun):
                pixels = _lattice(dimensions[:2])
                pixels["x"] = pixels["x"].to_numpy(dtype=float) + rng.normal(
                    0, nx * jitter_scale, size=len(pixels)
                )
                pixels["y"] = pixels["y"].to_numpy(dtype=float) + rng.normal(
                    0, ny * jitter_scale, size=len(pixels)
                )
                run_label = f"run{condition}{run_index + 1}"
                pixels["run"] = run_label
                if preset_number == 4:
                    region_name = f"circle{condition}"
                    pixels = add_shape(pixels, center, radius, name=region_name)
                else:
                    circle_center = {"x": nx / 4, "y": ny / 4}
                    square_center = {"x": 3 * nx / 4, "y": 3 * ny / 4}
                    pixels = add_shape(
                        pixels,
                        circle_center,
                        max(1, min(nx, ny) / 4),
                        shape="circle",
                        name=f"circle{condition}",
                    )
                    pixels = add_shape(
                        pixels,
                        square_center,
                        max(1, min(nx, ny) / 4),
                        shape="square",
                        name=f"square{condition}",
                    )
                pixels["condition"] = condition
                frames.append(pixels)
        pixel_data = PositionDataFrame(pd.concat(frames, ignore_index=True))
    else:
        run_frames = []
        design_repeats = 1 if preset_number == 9 else nrun
        for run_index in range(design_repeats):
            pixels = base.copy()
            pixels["run"] = f"run{run_index}"
            if preset_number == 1:
                pixels = add_shape(pixels, center, radius, name="circle")
            elif preset_number == 2:
                pixels = add_shape(
                    pixels, {"x": nx / 4, "y": ny / 4}, radius, name="circle"
                )
                pixels = add_shape(
                    pixels,
                    {"x": 3 * nx / 4, "y": 3 * ny / 4},
                    radius,
                    shape="square",
                    name="square",
                )
            elif preset_number == 3:
                pixels = add_shape(
                    pixels,
                    {"x": nx / 4, "y": ny / 4},
                    radius,
                    shape="square",
                    name="square1",
                )
                pixels = add_shape(
                    pixels,
                    {"x": 3 * nx / 4, "y": 3 * ny / 4},
                    radius,
                    shape="square",
                    name="square2",
                )
                pixels = add_shape(pixels, center, radius, name="circle")
            elif preset_number == 6:
                pixels = add_shape(
                    pixels,
                    {"x": nx / 4, "y": ny / 4},
                    radius,
                    shape="square",
                    name="square1",
                )
                pixels = add_shape(
                    pixels,
                    {"x": 3 * nx / 4, "y": 3 * ny / 4},
                    radius,
                    shape="square",
                    name="square2",
                )
                pixels = add_shape(pixels, center, radius, name="circleA")
                pixels["circleB"] = False
            elif preset_number in {7, 8}:
                if preset_number == 7:
                    size = radius
                    shape = "circle"
                else:
                    size = max(1, radius * 1.5)
                    shape = "square"
                pixels = add_shape(
                    pixels,
                    {"x": nx / 4, "y": ny / 4},
                    size,
                    shape=shape,
                    name=f"{shape}A" if preset_number == 7 else "squareA",
                )
                pixels = add_shape(
                    pixels,
                    {"x": 3 * nx / 4, "y": 3 * ny / 4},
                    size,
                    shape=shape,
                    name=f"{shape}B" if preset_number == 7 else "squareB",
                )
                if preset_number == 8:
                    pixels = add_shape(
                        pixels,
                        {"x": nx / 4, "y": ny / 4},
                        radius / 2,
                        shape="circle",
                        name="circleA",
                    )
                    pixels = add_shape(
                        pixels,
                        {"x": 3 * nx / 4, "y": 3 * ny / 4},
                        radius / 2,
                        shape="circle",
                        name="circleB",
                    )
                pixels["ref"] = True
                pixels["condition"] = np.where(
                    (pixels["x"] <= nx / 2) & (pixels["y"] <= ny / 2), "A", "B"
                )
            elif preset_number == 9:
                pixels = add_shape(pixels, center, radius, name="sphere1")
                pixels = add_shape(pixels, center, radius / 2, name="sphere2")
            run_frames.append(pixels)
        pixel_data = PositionDataFrame(pd.concat(run_frames, ignore_index=True))

    mz_values = np.sort(rng.lognormal(mean=7, sigma=0.3, size=npeaks))
    height_values = np.broadcast_to(
        np.asarray(peakheight, dtype=float), (len(np.atleast_1d(peakheight)),)
    )
    diff_values = np.broadcast_to(
        np.asarray(peakdiff, dtype=float), (len(np.atleast_1d(peakdiff)),)
    )
    features: dict[str, Any] = {"mz": mz_values}

    def _assign_region(
        name: str, active: NDArray[np.bool_], height_index: int = 0
    ) -> NDArray[np.float64]:
        mean = float(height_values[height_index % len(height_values)])
        active_indices = np.flatnonzero(active)
        return _positive_heights(rng, active_indices, mean, sdsample, npeaks)

    first = np.arange(npeaks) < int(np.ceil(npeaks / 3))
    middle = (np.arange(npeaks) >= npeaks // 3) & (np.arange(npeaks) < 2 * npeaks // 3)
    last = np.arange(npeaks) >= 2 * npeaks // 3
    if preset_number == 1:
        features["circle"] = _assign_region("circle", np.ones(npeaks, dtype=bool))
    elif preset_number == 2:
        features["circle"] = _assign_region("circle", ~last)
        features["square"] = _assign_region("square", ~first, 1)
    elif preset_number == 3:
        features["square1"] = _assign_region("square1", ~last)
        features["square2"] = _assign_region("square2", ~first, 1)
        features["circle"] = _assign_region("circle", middle, 2)
    elif preset_number in {4, 5}:
        if preset_number == 4:
            base_heights = _assign_region("circle", np.ones(npeaks, dtype=bool))
            difference = np.zeros(npeaks)
            difference[first] = float(diff_values[0])
            features["circleA"] = base_heights
            features["circleB"] = base_heights + difference
            features["diff"] = first
        else:
            circle_heights = _assign_region("circle", ~last)
            square_heights = _assign_region("square", ~first, 1)
            circle_difference = np.zeros(npeaks)
            square_difference = np.zeros(npeaks)
            circle_difference[first] = float(diff_values[0])
            square_difference[last] = float(diff_values[-1])
            features["circleA"] = circle_heights
            features["circleB"] = circle_heights + circle_difference
            features["squareA"] = square_heights
            features["squareB"] = square_heights + square_difference
            features["diff.circle"] = first
            features["diff.square"] = last
    elif preset_number == 6:
        features["square1"] = _assign_region("square1", ~last)
        features["square2"] = _assign_region("square2", ~first, 1)
        circle_heights = _assign_region("circle", middle, 2)
        circle_difference = np.zeros(npeaks)
        circle_difference[middle] = float(diff_values[0])
        features["circleA"] = circle_heights
        features["circleB"] = circle_heights + circle_difference
        features["diff.circle"] = middle
    elif preset_number in {7, 8}:
        features["ref"] = _assign_region("reference", last, 1)
        first_region = "circle" if preset_number == 7 else "square"
        if preset_number == 8:
            features["squareA"] = _assign_region("squareA", ~last)
            features["squareB"] = _assign_region("squareB", ~first, 1)
            first_region = "circle"
        region_heights = _assign_region(first_region, ~last, 1)
        region_difference = np.zeros(npeaks)
        region_difference[first] = float(diff_values[0])
        features[f"{first_region}A"] = region_heights
        features[f"{first_region}B"] = region_heights + region_difference
        features["diff"] = first
    else:
        features["sphere1"] = _assign_region("sphere1", ~last)
        features["sphere2"] = _assign_region("sphere2", ~first, 1)

    feature_data = MassDataFrame(features)
    return {"pixel_data": pixel_data, "feature_data": feature_data}


def _spatial_noise(
    coordinates: NDArray[np.float64],
    rho: float,
    rng: np.random.Generator,
    sar: bool,
) -> NDArray[np.float64]:
    count = len(coordinates)
    white = rng.normal(size=count)
    if count < 2 or rho == 0:
        return white
    pairs = cKDTree(coordinates).query_pairs(r=1.01, output_type="ndarray")
    if not len(pairs):
        return white
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
    adjacency = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(count, count)
    ).tocsr()
    row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sums)
    nonzero = row_sums > 0
    inverse[nonzero] = 1.0 / row_sums[nonzero]
    weights = sparse.diags(inverse) @ adjacency
    if sar:
        rho_stable = min(rho, 0.95)
        noise = spsolve(eye(count, format="csr") - rho_stable * weights, white)
    else:
        noise = (1 - rho) * white + rho * (weights @ white)
    standard_deviation = float(np.std(noise))
    return noise / standard_deviation if standard_deviation else white


def simulate_image(
    pixel_data: PositionDataFrame | pd.DataFrame | None = None,
    feature_data: MassDataFrame | pd.DataFrame | None = None,
    preset: int | None = None,
    *,
    from_: float | None = None,
    to: float | None = None,
    by: float = 400,
    sdrun: float = 1.0,
    sdpixel: float = 1.0,
    spcorr: float = 0.3,
    sar: bool = False,
    resolution: float = 1000,
    fmax: float = 0.5,
    units: str = "ppm",
    centroided: bool = False,
    continuous: bool = True,
    random_state: int | np.random.Generator | None = None,
    **preset_kwargs: Any,
) -> MSImagingArrays | MSImagingExperiment:
    """Simulate a complete imaging experiment from a design or preset.

    Matching boolean columns in ``pixel_data`` and numeric columns in
    ``feature_data`` define where each feature is present and its mean signal.
    If ``preset`` is supplied, ``preset_image_def`` creates that design.
    ``random_state`` accepts an integer seed or a NumPy ``Generator``.
    """
    if preset is not None:
        design = preset_image_def(preset, random_state=random_state, **preset_kwargs)
        pixel_data = design["pixel_data"]
        feature_data = design["feature_data"]
    if pixel_data is None or feature_data is None:
        raise ValueError("provide preset or both pixel_data and feature_data")
    pixels = PositionDataFrame(pixel_data)
    features = MassDataFrame(feature_data)
    if preset is not None and (from_ is not None or to is not None):
        lower = 0.9 * float(features.mz.min()) if from_ is None else float(from_)
        upper = 1.1 * float(features.mz.max()) if to is None else float(to)
        if not np.isfinite([lower, upper]).all() or lower <= 0 or upper <= lower:
            raise ValueError("from_ and to must define a positive increasing range")
        source_mz = features.mz
        span = float(np.ptp(source_mz))
        normalized = (
            np.zeros_like(source_mz)
            if span == 0
            else (source_mz - source_mz.min()) / span
        )
        shifted_features = pd.DataFrame(features).copy()
        shifted_features["mz"] = lower + (0.1 + 0.8 * normalized) * (upper - lower)
        features = MassDataFrame(shifted_features)
    shared_columns = [
        name
        for name in pixels.columns
        if name in features.columns and name not in {"mz", "run"}
    ]
    if not shared_columns:
        raise ValueError("pixel_data and feature_data need matching region columns")
    if not np.isfinite(spcorr) or not 0 <= spcorr < 1:
        raise ValueError("spcorr must be in [0, 1)")
    if not np.isfinite([sdrun, sdpixel]).all() or sdrun < 0 or sdpixel < 0:
        raise ValueError("sdrun and sdpixel must be nonnegative")
    rng = _rng(random_state)
    peak_mz = features.mz
    n_pixels = len(pixels)
    means = np.zeros((len(features), n_pixels), dtype=float)
    for column in shared_columns:
        mask = pixels[column].to_numpy(dtype=bool)
        values = pd.to_numeric(features[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(
                f"feature design column {column!r} must be finite and nonnegative"
            )
        means += values[:, None] * mask[None, :]
    run_categories = pixels.run
    run_effects = {run: rng.normal(0, sdrun) for run in run_categories.categories}
    run_error = np.array([run_effects[run] for run in run_categories], dtype=float)
    coordinates = pixels.coord.to_numpy(dtype=float)
    spatial_error = np.zeros(n_pixels, dtype=float)
    for run in run_categories.categories:
        selected = np.flatnonzero(np.asarray(run_categories == run))
        if selected.size:
            spatial_error[selected] = _spatial_noise(
                coordinates[selected], spcorr, rng, sar
            )
    pixel_error = sdpixel * spatial_error
    means = np.maximum(0, means + (run_error + pixel_error)[None, :])

    lower = 0.9 * float(peak_mz.min()) if from_ is None else float(from_)
    upper = 1.1 * float(peak_mz.max()) if to is None else float(to)
    domain = _mass_domain(lower, upper, by, units)
    spectra = _simulate_profiles(
        peak_mz,
        means,
        domain,
        sdpeaks=None,
        sdpeakmult=0.2,
        sdnoise=0.1,
        sdmz=10,
        resolution=resolution,
        fmax=fmax,
        baseline=0,
        decay=10,
        units=units,
        centroided=centroided,
        rng=rng,
    )
    metadata = {
        "design": {"pixel_data": pixels.copy(), "feature_data": features.copy()},
        "random_state": random_state if isinstance(random_state, int) else None,
    }
    if continuous:
        if centroided:
            mass_axis = peak_mz
            matrix = spectra
        else:
            mass_axis = domain
            matrix = spectra
        return MSImagingExperiment(
            matrix,
            feature_data=MassDataFrame(mz=mass_axis),
            pixel_data=pixels,
            centroided=centroided,
            metadata=metadata,
        )
    if centroided:
        mz_vectors = []
        intensity_vectors = []
        for pixel_index in range(n_pixels):
            shifts = rng.normal(0, 10, len(peak_mz))
            observed_mz = (
                peak_mz * (1 + shifts * 1e-6) if units == "ppm" else peak_mz + shifts
            )
            order = np.argsort(observed_mz)
            mz_vectors.append(observed_mz[order])
            intensity_vectors.append(spectra[:, pixel_index][order])
    else:
        mz_vectors = [domain.copy() for _ in range(n_pixels)]
        intensity_vectors = [spectra[:, pixel_index] for pixel_index in range(n_pixels)]
    return MSImagingArrays(
        mz=mz_vectors,
        intensity=intensity_vectors,
        pixel_data=pixels,
        centroided=centroided,
        continuous=False,
        metadata=metadata,
    )
