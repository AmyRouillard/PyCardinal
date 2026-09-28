"""Peak detection, reference alignment, and m/z-domain utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, find_peaks_cwt, peak_widths

from pycardinal.core import (
    MSImagingArrays,
    MSImagingExperiment,
    SpectralImagingData,
    add_processing,
    process,
)
from pycardinal.core.metadata import MassDataFrame
from pycardinal.io import convert_arrays_to_experiment, convert_experiment_to_arrays


def _noise(signal: NDArray[np.float64], method: str) -> float:
    if signal.size < 2:
        return 0.0
    if method == "sd":
        return float(np.std(signal))
    if method == "mad":
        center = np.median(signal)
        return float(1.4826 * np.median(np.abs(signal - center)))
    difference = np.diff(signal)
    if method == "quantile":
        return float(np.quantile(np.abs(difference), 0.75))
    if method == "filter":
        return float(np.median(np.abs(difference)))
    return float(
        1.4826 * np.median(np.abs(difference - np.median(difference))) / np.sqrt(2)
    )


def _peak_summary(
    signal: NDArray[np.float64],
    index: int,
    type_: str,
    domain: NDArray[np.float64],
) -> float:
    if type_ == "height":
        return float(signal[index])
    if type_ != "area":
        raise ValueError("type_ must be 'height' or 'area'")
    if len(signal) < 2:
        return float(signal[index])
    widths = peak_widths(signal, [index], rel_height=0.5)
    left = max(0, int(np.floor(widths[2][0])))
    right = min(len(signal) - 1, int(np.ceil(widths[3][0])))
    if right <= left:
        return float(signal[index])
    return float(trapezoid(signal[left : right + 1], domain[left : right + 1]))


def _detect_peaks(
    signal: NDArray[np.float64],
    domain: NDArray[np.float64],
    method: str,
    snr: float,
    type_: str,
    **options: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(signal) != len(domain):
        raise ValueError("intensity and m/z must have equal lengths")
    if len(signal) < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    noise = _noise(signal, method)
    threshold = snr * noise
    if method == "cwt":
        max_width = min(int(options.get("max_width", 8)), max(1, len(signal) // 3))
        indices = find_peaks_cwt(signal, np.arange(1, max_width + 1))
        peak_indices = np.asarray(indices, dtype=int)
        peak_indices = peak_indices[
            (peak_indices > 0) & (peak_indices < len(signal) - 1)
        ]
    else:
        kwargs: dict[str, Any] = {"height": threshold}
        if method == "filter":
            kwargs["prominence"] = threshold
        if "distance" in options:
            kwargs["distance"] = options["distance"]
        peak_indices, _ = find_peaks(signal, **kwargs)
    peak_indices = np.unique(peak_indices)
    peak_indices = peak_indices[signal[peak_indices] > 0]
    if not len(peak_indices):
        return np.array([], dtype=float), np.array([], dtype=float)
    values = np.asarray(
        [_peak_summary(signal, int(index), type_, domain) for index in peak_indices],
        dtype=float,
    )
    return domain[peak_indices].astype(float), values


def _reference_pick(
    signal: NDArray[np.float64],
    domain: NDArray[np.float64],
    ref: NDArray[np.float64],
    tolerance: float | None,
    units: str,
    type_: str,
) -> NDArray[np.float64]:
    result = np.zeros(len(ref), dtype=float)
    peak_indices, _ = find_peaks(signal)
    if not len(peak_indices):
        return result
    peak_mz = domain[peak_indices]
    for ref_index, target in enumerate(ref):
        distances = np.abs(peak_mz - target)
        if units == "ppm":
            distances = distances / target * 1e6
        closest = int(np.argmin(distances))
        if tolerance is not None and distances[closest] > tolerance:
            continue
        result[ref_index] = _peak_summary(
            signal,
            int(peak_indices[closest]),
            type_,
            domain,
        )
    return result


def _pick_spectrum(
    intensity: ArrayLike,
    mz: ArrayLike,
    *,
    method: str,
    snr: float,
    type_: str,
    ref: NDArray[np.float64] | None,
    tolerance: float | None,
    units: str,
    options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(intensity, dtype=float)
    domain = np.asarray(mz, dtype=float)
    if ref is not None:
        return ref.copy(), _reference_pick(signal, domain, ref, tolerance, units, type_)
    peaks, values = _detect_peaks(signal, domain, method, snr, type_, **options)
    return peaks, values


def _pick_shared_spectrum(
    intensity: ArrayLike,
    mz: ArrayLike,
    *,
    method: str,
    snr: float,
    type_: str,
    options: dict[str, Any],
) -> NDArray[np.float64]:
    signal = np.asarray(intensity, dtype=float)
    domain = np.asarray(mz, dtype=float)
    peak_mz, peak_values = _detect_peaks(signal, domain, method, snr, type_, **options)
    result = np.zeros_like(signal)
    if len(peak_mz):
        indices = np.searchsorted(domain, peak_mz).clip(0, len(domain) - 1)
        result[indices] = peak_values
    return result


def peak_pick(
    obj: SpectralImagingData,
    ref: ArrayLike | MassDataFrame | None = None,
    method: str = "diff",
    snr: float = 2.0,
    type_: str = "height",
    tolerance: float | None = None,
    units: str = "ppm",
    **options: Any,
) -> SpectralImagingData:
    """Queue local-maximum peak picking, optionally extracting a reference.

    Methods ``diff``, ``sd``, ``mad``, ``quantile``, ``filter``, and ``cwt``
    select noise estimates/detection backends. ``type_`` is ``height`` or
    ``area``. Without ``ref``, ragged spectra become per-spectrum peak lists;
    shared-axis experiments keep the shared axis and zero non-peak samples.
    """
    methods = {"diff", "sd", "mad", "quantile", "filter", "cwt"}
    if method not in methods:
        raise ValueError(f"method must be one of {sorted(methods)}")
    if type_ not in {"height", "area"}:
        raise ValueError("type_ must be 'height' or 'area'")
    if not np.isfinite(snr) or snr < 0:
        raise ValueError("snr must be finite and nonnegative")
    if units not in {"ppm", "mz"}:
        raise ValueError("units must be 'ppm' or 'mz'")
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be finite and nonnegative")
    reference = ref.mz if isinstance(ref, MassDataFrame) else ref
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 1 or not np.isfinite(reference).all():
            raise ValueError("ref must be a finite one-dimensional vector")
        reference = np.unique(reference)
        if tolerance is None and len(reference) > 1:
            reference_gaps = np.diff(reference)
            tolerance = float(np.min(reference_gaps) / 2)
            if units == "ppm":
                tolerance = tolerance / float(np.min(reference)) * 1e6
    callback = (
        _pick_shared_spectrum
        if isinstance(obj, MSImagingExperiment) and reference is None
        else _pick_spectrum
    )
    arguments: dict[str, Any] = {
        "method": method,
        "snr": snr,
        "type_": type_,
        "options": dict(options),
    }
    if callback is _pick_spectrum:
        arguments.update({"ref": reference, "tolerance": tolerance, "units": units})
    queued = add_processing(
        obj,
        callback,
        label=f"{type_} peak picking",
        metadata={"method": method, "snr": snr, "type": type_},
        **arguments,
    )
    if isinstance(queued, (MSImagingArrays, MSImagingExperiment)):
        queued.centroided = True
    return queued


def _as_mz_vector(ref: ArrayLike | MassDataFrame | None) -> NDArray[np.float64] | None:
    if ref is None:
        return None
    values = ref.mz if isinstance(ref, MassDataFrame) else ref
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 1
        or not np.isfinite(result).all()
        or np.any(np.diff(result) <= 0)
    ):
        raise ValueError("ref must be a finite, strictly increasing m/z vector")
    return result


def _distance_matrix(
    observed: NDArray[np.float64],
    reference: NDArray[np.float64],
    units: str,
) -> NDArray[np.float64]:
    distance = np.abs(observed[:, None] - reference[None, :])
    if units == "ppm":
        distance = distance / observed[:, None] * 1e6
    return distance


def _merge_reference(
    values: NDArray[np.float64],
    tolerance: float,
    units: str,
) -> NDArray[np.float64]:
    if not len(values):
        return np.array([], dtype=float)
    ordered = np.sort(values)
    clusters: list[list[float]] = [[float(ordered[0])]]
    for value in ordered[1:]:
        anchor = float(np.mean(clusters[-1]))
        distance = abs(float(value) - anchor)
        if units == "ppm":
            distance = distance / anchor * 1e6
        if distance <= tolerance:
            clusters[-1].append(float(value))
        else:
            clusters.append([float(value)])
    return np.asarray([np.mean(cluster) for cluster in clusters], dtype=float)


def peak_align(
    obj: SpectralImagingData,
    ref: ArrayLike | MassDataFrame | None = None,
    *,
    method: str = "diff",
    snr: float = 2.0,
    tolerance: float | None = None,
    units: str = "ppm",
    binratio: float = 2.0,
    n_jobs: int | None = None,
) -> MSImagingExperiment:
    """Apply queued steps and align detected peaks to a shared sparse matrix.

    When ``ref`` is omitted, local maxima are detected, merged by tolerance,
    and used as the reference axis. This is an eager operation and returns
    centroided feature-by-pixel data.
    """
    if not isinstance(obj, (MSImagingArrays, MSImagingExperiment)):
        raise TypeError("peak_align requires MSImagingArrays or MSImagingExperiment")
    if units not in {"ppm", "mz"}:
        raise ValueError("units must be 'ppm' or 'mz'")
    if not np.isfinite(binratio) or binratio <= 0:
        raise ValueError("binratio must be positive and finite")
    if obj.processing:
        obj = cast(Any, process(obj, n_jobs=n_jobs))
    aligned_obj = cast(MSImagingArrays | MSImagingExperiment, obj)
    if isinstance(aligned_obj, MSImagingArrays):
        pairs = list(zip(aligned_obj.mz, aligned_obj.intensity))
    else:
        matrix = aligned_obj.intensity
        pairs = [
            (
                aligned_obj.mz,
                (
                    np.asarray(cast(Any, matrix).getcol(index).toarray()).ravel()
                    if sparse.issparse(matrix)
                    else np.asarray(matrix)[:, index]
                ),
            )
            for index in range(aligned_obj.shape[1])
        ]
    reference = _as_mz_vector(ref)
    if reference is None:
        detected = [
            _detect_peaks(np.asarray(signal), np.asarray(masses), method, snr, "height")
            for masses, signal in pairs
        ]
    else:
        detected = [
            (np.asarray(masses, dtype=float), np.asarray(signal, dtype=float))
            for masses, signal in pairs
        ]
    if tolerance is None:
        if reference is not None and len(reference) > 1:
            gap = np.diff(reference)
            tolerance = float(np.median(gap) * binratio / 2)
            if units == "ppm":
                tolerance = tolerance / float(np.median(reference)) * 1e6
        else:
            intra_gaps = [np.diff(peaks) for peaks, _ in detected if len(peaks) > 1]
            spacing = np.concatenate(intra_gaps) if intra_gaps else np.array([0.0])
            raw_tolerance = float(np.min(spacing) / binratio)
            if units == "ppm" and raw_tolerance:
                centers = np.concatenate([peaks for peaks, _ in detected if len(peaks)])
                tolerance = raw_tolerance / float(np.median(centers)) * 1e6
            else:
                tolerance = raw_tolerance
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if reference is None:
        all_peaks = (
            np.concatenate([peaks for peaks, _ in detected if len(peaks)])
            if any(len(peaks) for peaks, _ in detected)
            else np.array([], dtype=float)
        )
        reference = _merge_reference(all_peaks, tolerance, units)
    if not len(reference):
        raise ValueError("no peaks detected; adjust method/SNR or supply ref")

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    counts = np.zeros(len(reference), dtype=np.int64)
    for pixel_index, (peak_mz, peak_values) in enumerate(detected):
        if not len(peak_mz):
            continue
        distances = _distance_matrix(peak_mz, reference, units)
        feature_indices = np.argmin(distances, axis=1)
        minimum = distances[np.arange(len(peak_mz)), feature_indices]
        matched = minimum <= tolerance
        if np.any(matched):
            rows = feature_indices[matched]
            row_parts.append(rows.astype(np.int64))
            col_parts.append(
                np.full(np.count_nonzero(matched), pixel_index, dtype=np.int64)
            )
            data_parts.append(peak_values[matched])
            counts += np.bincount(rows, minlength=len(reference))
    intensity = sparse.coo_matrix(
        (
            np.concatenate(data_parts) if data_parts else np.array([], dtype=float),
            (
                (
                    np.concatenate(row_parts)
                    if row_parts
                    else np.array([], dtype=np.int64)
                ),
                (
                    np.concatenate(col_parts)
                    if col_parts
                    else np.array([], dtype=np.int64)
                ),
            ),
        ),
        shape=(len(reference), len(aligned_obj)),
    ).tocsr()
    feature_data = MassDataFrame(
        mz=reference,
        count=counts,
        freq=counts / max(len(aligned_obj), 1),
    )
    return MSImagingExperiment(
        {"intensity": intensity},
        feature_data=feature_data,
        pixel_data=aligned_obj.pixel_data,
        experiment_data=aligned_obj.experiment_data,
        centroided=True,
        metadata=aligned_obj.metadata,
    )


def peak_process(
    obj: SpectralImagingData,
    ref: ArrayLike | MassDataFrame | None = None,
    *,
    method: str = "diff",
    snr: float = 2.0,
    type_: str = "height",
    tolerance: float | None = None,
    units: str = "ppm",
    sample_size: int | float | None = None,
    binratio: float = 2.0,
    filter_freq: bool | int | float = True,
    n_jobs: int | None = None,
    **peak_options: Any,
) -> MSImagingExperiment:
    """Convenience pipeline for peak picking followed by peak alignment.

    If ``sample_size`` is set and ``ref`` is omitted, evenly spaced spectra
    are peak-picked to estimate a reference before extracting peaks from the
    full dataset. A fraction in ``(0, 1)`` is treated as a proportion; values
    at least 1 are interpreted as a count.
    """
    if sample_size is not None and ref is None:
        if len(obj) == 0:
            raise ValueError("cannot estimate a peak reference from an empty dataset")
        if not np.isfinite(sample_size) or sample_size <= 0:
            raise ValueError("sample_size must be positive and finite")
        if not np.isfinite(binratio) or binratio <= 0:
            raise ValueError("binratio must be positive and finite")
        count = (
            int(np.ceil(sample_size * len(obj)))
            if sample_size < 1
            else min(int(sample_size), len(obj))
        )
        indices = np.unique(np.linspace(0, len(obj) - 1, count, dtype=int))
        sample: MSImagingArrays | MSImagingExperiment
        if isinstance(obj, MSImagingArrays):
            sample = MSImagingArrays(
                mz=[obj.mz[index] for index in indices],
                intensity=[obj.intensity[index] for index in indices],
                pixel_data=obj.pixel_data.iloc[indices].reset_index(drop=True),
                experiment_data=obj.experiment_data,
                centroided=obj.centroided,
                metadata=obj.metadata,
            )
        elif isinstance(obj, MSImagingExperiment):
            sample = obj[:, indices]
        else:
            raise TypeError("peak_process requires mass-imaging data")
        sampled = cast(
            MSImagingArrays | MSImagingExperiment,
            process(
                peak_pick(
                    sample,
                    method=method,
                    snr=snr,
                    type_=type_,
                    **peak_options,
                ),
                n_jobs=n_jobs,
            ),
        )
        if isinstance(sampled, MSImagingArrays):
            locations = [np.asarray(masses, dtype=float) for masses in sampled.mz]
        else:
            matrix = sampled.intensity
            locations = [
                _detect_peaks(
                    (
                        np.asarray(cast(Any, matrix).getcol(index).toarray()).ravel()
                        if sparse.issparse(matrix)
                        else np.asarray(matrix)[:, index]
                    ),
                    sampled.mz,
                    method,
                    snr,
                    "height",
                )[0]
                for index in range(sampled.shape[1])
            ]
        observed = (
            np.concatenate([peaks for peaks in locations if len(peaks)])
            if any(len(peaks) for peaks in locations)
            else np.array([], dtype=float)
        )
        if tolerance is None:
            gaps = [np.diff(np.sort(peaks)) for peaks in locations if len(peaks) > 1]
            all_gaps = np.concatenate(gaps) if gaps else np.array([], dtype=float)
            raw_tolerance = float(np.min(all_gaps) / binratio) if len(all_gaps) else 0.0
            if units == "ppm" and raw_tolerance and len(observed):
                tolerance = raw_tolerance / float(np.median(observed)) * 1e6
            else:
                tolerance = raw_tolerance
        ref = _merge_reference(observed, tolerance, units)

    picked = process(
        peak_pick(
            obj,
            ref=ref,
            method=method,
            snr=snr,
            type_=type_,
            tolerance=tolerance,
            units=units,
            **peak_options,
        ),
        n_jobs=n_jobs,
    )
    aligned = peak_align(
        picked,
        ref=ref,
        method=method,
        snr=snr,
        tolerance=tolerance,
        units=units,
        n_jobs=n_jobs,
    )
    if filter_freq is False or filter_freq == 0:
        return aligned
    if isinstance(filter_freq, bool):
        minimum = 1
    elif isinstance(filter_freq, float) and filter_freq < 1:
        minimum = int(np.ceil(filter_freq * len(aligned)))
    else:
        minimum = int(filter_freq)
    if minimum < 0:
        raise ValueError("filter_freq must be nonnegative")
    keep = np.asarray(aligned.feature_data["count"] > minimum)
    return aligned[keep, :]


def estimate_domain(
    xlist: Sequence[ArrayLike],
    width: str = "median",
    units: str = "relative",
) -> NDArray[np.float64]:
    """Estimate a regular shared domain from per-spectrum m/z vectors."""
    if width not in {"median", "min", "max", "mean"}:
        raise ValueError("width must be median, min, max, or mean")
    if units not in {"relative", "absolute"}:
        raise ValueError("units must be 'relative' or 'absolute'")
    sorted_vectors = [np.sort(np.asarray(values, dtype=float)) for values in xlist]
    sorted_vectors = [
        values[np.isfinite(values)] for values in sorted_vectors if len(values)
    ]
    if not sorted_vectors:
        raise ValueError("xlist must contain at least one non-empty domain")
    low = min(float(values[0]) for values in sorted_vectors)
    high = max(float(values[-1]) for values in sorted_vectors)
    gaps = np.concatenate(
        [np.diff(values) for values in sorted_vectors if len(values) > 1]
    )
    gaps = gaps[gaps > 0]
    if not len(gaps):
        return np.asarray([low], dtype=float)
    step = float(getattr(np, width)(gaps))
    if units == "relative":
        step /= float(np.median(np.concatenate(sorted_vectors)))
        step *= max(abs(low), 1.0)
    if step <= 0:
        raise ValueError("could not estimate a positive domain spacing")
    count = min(int(np.floor((high - low) / step)) + 1, 2_000_000)
    return low + np.arange(count, dtype=float) * step


def estimate_reference_mz(
    obj: MSImagingArrays | MSImagingExperiment,
    width: str = "median",
    units: str = "ppm",
) -> NDArray[np.float64]:
    """Return the shared m/z axis or estimate one from ragged spectra."""
    if isinstance(obj, MSImagingExperiment):
        return obj.mz.copy()
    normalized_units = (
        "relative" if units == "ppm" else "absolute" if units == "mz" else units
    )
    return estimate_domain(obj.mz, width=width, units=normalized_units)


def estimate_reference_peaks(
    obj: SpectralImagingData,
    method: str = "diff",
    snr: float = 2.0,
) -> MassDataFrame:
    """Detect reference peaks from the mean spectrum of a dataset."""
    if isinstance(obj, MSImagingArrays):
        reference_mz = estimate_reference_mz(obj, units="mz")
        shared = convert_arrays_to_experiment(
            obj,
            mz=reference_mz,
            units="mz",
            tolerance=(
                float(np.median(np.diff(reference_mz)) / 2)
                if len(reference_mz) > 1
                else 0.0
            ),
        )
        matrix = shared.intensity
    elif isinstance(obj, MSImagingExperiment):
        reference_mz = obj.mz
        matrix = obj.intensity
    else:
        raise TypeError("estimate_reference_peaks requires mass-imaging data")
    mean_signal = (
        np.asarray(cast(Any, matrix).mean(axis=1)).ravel()
        if sparse.issparse(matrix)
        else np.asarray(matrix, dtype=float).mean(axis=1)
    )
    peak_mz, peak_intensity = _detect_peaks(
        mean_signal,
        reference_mz,
        method,
        snr,
        "height",
    )
    return MassDataFrame({"mz": peak_mz, "intensity": peak_intensity})


def bin_spectra(
    obj: MSImagingArrays | MSImagingExperiment,
    ref: ArrayLike | MassDataFrame | None = None,
    *,
    method: str = "sum",
    resolution: float | None = None,
    tolerance: float | None = None,
    units: str = "ppm",
    mass_range: tuple[float, float] | None = None,
) -> MSImagingExperiment:
    """Eagerly bin spectra onto shared m/z values.

    Aggregation methods ``sum``, ``mean``, ``max``, and ``min`` are supported.
    Interpolation names from the R API are recognized but not yet implemented.
    """
    if method not in {"sum", "mean", "max", "min"}:
        if method in {"linear", "cubic", "gaussian", "lanczos"}:
            raise NotImplementedError(
                f"interpolation method {method!r} is not implemented"
            )
        raise ValueError("unsupported bin aggregation method")
    reference = _as_mz_vector(ref)
    arrays = (
        convert_experiment_to_arrays(obj)
        if isinstance(obj, MSImagingExperiment)
        else obj
    )
    if reference is None:
        if resolution is None:
            raise ValueError("provide ref or resolution for binning")
        inferred_range = mass_range or _mass_bounds(arrays, resolution, units)
        reference = _build_reference_axis(inferred_range, resolution, units)
    bin_tolerance = tolerance
    if bin_tolerance is None:
        if len(reference) > 1:
            spacing = float(np.median(np.diff(reference)))
            bin_tolerance = (
                spacing / float(np.median(reference)) * 1e6 / 2
                if units == "ppm"
                else spacing / 2
            )
        else:
            bin_tolerance = 0.0
    result = convert_arrays_to_experiment(
        arrays,
        mz=reference,
        mass_range=mass_range,
        units=units,
        tolerance=bin_tolerance,
    )
    if method == "sum":
        return result
    counts = _bin_counts(arrays, result.mz, bin_tolerance, units)
    matrix = result.intensity.toarray()
    if method == "mean":
        matrix = np.divide(matrix, counts, out=np.zeros_like(matrix), where=counts > 0)
    elif method == "max":
        matrix = _aggregate_duplicate_bins(
            arrays,
            result.mz,
            bin_tolerance,
            units,
            "max",
        )
    elif method == "min":
        matrix = _aggregate_duplicate_bins(
            arrays,
            result.mz,
            bin_tolerance,
            units,
            "min",
        )
    result.intensity = matrix
    return result


def _mass_bounds(
    obj: MSImagingArrays,
    resolution: float,
    units: str,
) -> tuple[float, float]:
    del resolution
    del units
    masses = np.concatenate(
        [np.asarray(values, dtype=float) for values in obj.mz if len(values)]
    )
    if not len(masses):
        raise ValueError("cannot infer mass range from empty spectra")
    return float(masses.min()), float(masses.max())


def _build_reference_axis(
    mass_range: tuple[float, float],
    resolution: float,
    units: str,
) -> NDArray[np.float64]:
    low, high = mass_range
    if (
        not np.isfinite([low, high, resolution]).all()
        or low <= 0
        or high < low
        or resolution <= 0
    ):
        raise ValueError("invalid mass_range or resolution")
    if units == "mz":
        count = int(np.floor((high - low) / resolution)) + 1
        return np.asarray(
            low + np.arange(min(count, 2_000_000), dtype=float) * resolution,
            dtype=float,
        )
    if units != "ppm":
        raise ValueError("units must be 'ppm' or 'mz'")
    ratio = 1 + resolution * 1e-6
    count = int(np.ceil(np.log(high / low) / np.log(ratio))) + 1
    return np.asarray(
        low * np.power(ratio, np.arange(min(count, 2_000_000), dtype=float)),
        dtype=float,
    )


def _bin_counts(
    obj: MSImagingArrays,
    reference: NDArray[np.float64],
    tolerance: float | None,
    units: str,
) -> NDArray[np.float64]:
    counts = np.zeros((len(reference), len(obj)), dtype=float)
    for pixel, masses in enumerate(obj.mz):
        masses = np.asarray(masses, dtype=float)
        if not len(masses):
            continue
        indices = np.searchsorted(reference, masses).clip(0, len(reference) - 1)
        left = np.maximum(indices - 1, 0)
        choose_left = np.abs(reference[left] - masses) < np.abs(
            reference[indices] - masses
        )
        indices = np.where(choose_left, left, indices)
        error = np.abs(reference[indices] - masses)
        if units == "ppm":
            error = error / masses * 1e6
        if tolerance is not None:
            indices = indices[error <= tolerance]
        np.add.at(counts[:, pixel], indices, 1)
    return counts


def _aggregate_duplicate_bins(
    obj: MSImagingArrays,
    reference: NDArray[np.float64],
    tolerance: float | None,
    units: str,
    operation: str,
) -> NDArray[np.float64]:
    result = np.zeros((len(reference), len(obj)), dtype=float)
    fills = np.zeros_like(result, dtype=bool)
    for pixel, (masses, signals) in enumerate(zip(obj.mz, obj.intensity)):
        masses = np.asarray(masses, dtype=float)
        signals = np.asarray(signals, dtype=float)
        if not len(masses):
            continue
        indices = np.searchsorted(reference, masses).clip(0, len(reference) - 1)
        left = np.maximum(indices - 1, 0)
        choose_left = np.abs(reference[left] - masses) < np.abs(
            reference[indices] - masses
        )
        indices = np.where(choose_left, left, indices)
        error = np.abs(reference[indices] - masses)
        if units == "ppm":
            error = error / masses * 1e6
        keep = (
            np.ones(len(indices), dtype=bool)
            if tolerance is None
            else error <= tolerance
        )
        for row, value in zip(indices[keep], signals[keep]):
            if fills[row, pixel]:
                result[row, pixel] = (
                    max(result[row, pixel], value)
                    if operation == "max"
                    else min(result[row, pixel], value)
                )
            else:
                result[row, pixel] = value
                fills[row, pixel] = True
    return result
