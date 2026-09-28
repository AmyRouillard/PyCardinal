"""Read, convert, and write imzML mass-spectrometry imaging datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pyimzml.ImzMLParser import ImzMLParser  # type: ignore[import-untyped]
from pyimzml.ImzMLWriter import ImzMLWriter  # type: ignore[import-untyped]
from scipy import sparse

from pycardinal.core import (
    MassDataFrame,
    MSImagingArrays,
    MSImagingExperiment,
    PositionDataFrame,
    SpectralImagingData,
)

_LOGGER = logging.getLogger(__name__)
_CONTINUOUS_ACCESSION = "IMS:1000030"
_PROCESSED_ACCESSION = "IMS:1000031"


def _representation(parser: ImzMLParser) -> str:
    for element in parser.root.iter():
        if not element.tag.endswith("cvParam"):
            continue
        if element.attrib.get("accession") == _CONTINUOUS_ACCESSION:
            return "continuous"
        if element.attrib.get("accession") == _PROCESSED_ACCESSION:
            return "processed"
    lengths = getattr(parser, "mzLengths", [])
    return "continuous" if lengths and len(set(lengths)) == 1 else "processed"


def _position_data(parser: ImzMLParser) -> PositionDataFrame:
    coordinates = parser.coordinates
    if not coordinates:
        return PositionDataFrame(np.empty((0, 2)))
    dimensions = len(coordinates[0])
    if dimensions not in (2, 3):
        raise ValueError(f"imzML pixel coordinates must be 2D or 3D, got {dimensions}D")
    names = ["x", "y", "z"][:dimensions]
    frame = pd.DataFrame(coordinates, columns=names)
    frame["run"] = "run1"
    return PositionDataFrame(frame)


def _metadata(parser: ImzMLParser, representation: str) -> dict[str, Any]:
    return {
        "source": str(parser.filename),
        "representation": representation,
        "spectrum_mode": parser.spectrum_mode,
        "imzml": dict(parser.imzmldict),
    }


def read_imzml(
    file: str | Path,
    *,
    memory: bool = True,
    check: bool = False,
    mass_range: tuple[float, float] | None = None,
    resolution: float | None = None,
    units: str = "ppm",
    guess_max: int = 1000,
    as_: str = "auto",
    parse_only: bool = False,
    verbose: bool = False,
) -> SpectralImagingData | dict[str, Any]:
    """Read an imzML file into a mass-imaging dataset.

    This first implementation reads spectra eagerly. ``memory=False`` is
    reserved for future file-backed loading and currently raises
    ``NotImplementedError`` rather than silently loading the full file.

    Parameters
    ----------
    file
        Path to the imzML XML file. Its matching ``.ibd`` sidecar must be
        present next to it.
    memory
        Must be ``True``; lazy/file-backed loading is not implemented yet.
    check
        If true, ask pyimzml to validate the imzML/ibd checksum where supported.
    mass_range
        Optional ``(minimum_mz, maximum_mz)`` used when converting processed
        spectra to a shared m/z axis.
    resolution
        Positive shared-axis spacing. Required with ``mass_range`` when
        converting processed spectra to an experiment unless ``mz`` is
        otherwise available through :func:`convert_arrays_to_experiment`.
    units
        ``"mz"`` for absolute spacing or ``"ppm"`` for parts-per-million.
    guess_max
        Number of evenly spaced spectra sampled to infer a mass range when
        conversion to a shared axis is requested.
    as_
        ``"auto"`` preserves processed spectra as ragged arrays and reads
        continuous data as an experiment. ``"arrays"`` and ``"experiment"``
        force a representation.
    parse_only
        Return parsed metadata and coordinates without reading spectra.
    verbose
        Emit a short progress message through the standard Python logger.

    Returns
    -------
    MSImagingArrays or MSImagingExperiment or dict
        A dataset, or parsed metadata when ``parse_only=True``.

    Raises
    ------
    FileNotFoundError
        If the XML or binary sidecar is absent.
    ValueError
        If ``as_`` or conversion parameters are invalid.
    NotImplementedError
        If ``memory=False`` is requested before file-backed loading exists.
    """
    path = Path(file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".imzml":
        raise ValueError(f"expected an .imzML file, got {path.name!r}")
    binary_path = path.with_suffix(".ibd")
    if not binary_path.is_file():
        raise FileNotFoundError(f"matching imzML binary file not found: {binary_path}")
    if not memory:
        raise NotImplementedError("file-backed imzML loading is not implemented yet")
    if check:
        raise NotImplementedError(
            "imzML/ibd checksum validation is not implemented yet"
        )
    if as_ not in {"auto", "arrays", "experiment"}:
        raise ValueError("as_ must be 'auto', 'arrays', or 'experiment'")
    if guess_max < 1:
        raise ValueError("guess_max must be positive")
    if units not in {"mz", "ppm"}:
        raise ValueError("units must be 'mz' or 'ppm'")

    if verbose:
        _LOGGER.info("Reading imzML file %s", path)
    parser = ImzMLParser(str(path))
    representation = _representation(parser)
    pixels = _position_data(parser)
    metadata = _metadata(parser, representation)
    if parse_only:
        return {
            **metadata,
            "coordinates": pixels.coord,
            "pixel_data": pixels,
            "centroided": parser.spectrum_mode == "centroid",
        }

    mz_spectra: list[NDArray[Any]] = []
    intensity_spectra: list[NDArray[Any]] = []
    if representation == "continuous":
        if not parser.coordinates:
            mz_axis = np.array([], dtype=float)
        else:
            mz_axis, _ = parser.getspectrum(0)
            mz_axis = np.asarray(mz_axis, dtype=float)
        matrix = np.empty((len(mz_axis), len(parser.coordinates)), dtype=float)
        for pixel_index in range(len(parser.coordinates)):
            _, values = parser.getspectrum(pixel_index)
            signal = np.asarray(values, dtype=float)
            if signal.size != mz_axis.size:
                raise ValueError("continuous imzML spectra have inconsistent lengths")
            matrix[:, pixel_index] = signal
        result: SpectralImagingData = MSImagingExperiment(
            matrix,
            feature_data=MassDataFrame(mz_axis),
            pixel_data=pixels,
            experiment_data=metadata,
            centroided=parser.spectrum_mode == "centroid",
        )
    else:
        for pixel_index in range(len(parser.coordinates)):
            mz_values, signal = parser.getspectrum(pixel_index)
            mz_spectra.append(np.asarray(mz_values, dtype=float))
            intensity_spectra.append(np.asarray(signal, dtype=float))
        result = MSImagingArrays(
            mz=mz_spectra,
            intensity=intensity_spectra,
            pixel_data=pixels,
            experiment_data=metadata,
            centroided=parser.spectrum_mode == "centroid",
            continuous=False,
        )

    if as_ == "arrays" and isinstance(result, MSImagingExperiment):
        result = convert_experiment_to_arrays(result)
    elif as_ == "experiment" and isinstance(result, MSImagingArrays):
        result = convert_arrays_to_experiment(
            result,
            mass_range=mass_range,
            resolution=resolution,
            units=units,
            guess_max=guess_max,
        )
    return result


def _sampled_mass_range(
    obj: MSImagingArrays,
    guess_max: int,
) -> tuple[float, float]:
    if len(obj) == 0:
        raise ValueError("cannot infer an m/z range from an empty dataset")
    sample_count = min(guess_max, len(obj))
    indices = np.unique(np.linspace(0, len(obj) - 1, sample_count, dtype=int))
    minima = [float(np.min(obj.mz[index])) for index in indices if len(obj.mz[index])]
    maxima = [float(np.max(obj.mz[index])) for index in indices if len(obj.mz[index])]
    if not minima:
        raise ValueError("cannot infer an m/z range from empty spectra")
    return min(minima), max(maxima)


def _make_mz_axis(
    mass_range: tuple[float, float],
    resolution: float,
    units: str,
) -> NDArray[np.float64]:
    lower, upper = map(float, mass_range)
    if not np.isfinite([lower, upper]).all() or lower <= 0 or upper < lower:
        raise ValueError("mass_range must contain finite positive increasing bounds")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a positive finite number")
    if units == "mz":
        bins = int(np.floor((upper - lower) / resolution)) + 1
        if bins > 5_000_000:
            raise ValueError("requested m/z axis exceeds 5 million bins")
        axis = lower + np.arange(bins, dtype=float) * resolution
        if axis[-1] < upper:
            axis = np.append(axis, upper)
        return axis
    if units != "ppm":
        raise ValueError("units must be 'mz' or 'ppm'")
    ratio = 1.0 + resolution * 1e-6
    if ratio <= 1:
        raise ValueError("ppm resolution must be greater than zero")
    count = int(np.ceil(np.log(upper / lower) / np.log(ratio))) + 1
    if count > 5_000_000:
        raise ValueError("requested m/z axis exceeds 5 million bins")
    axis = lower * np.power(ratio, np.arange(count, dtype=float))
    if axis[-1] < upper:
        axis = np.append(axis, upper)
    return axis


def convert_arrays_to_experiment(
    obj: MSImagingArrays,
    *,
    mz: ArrayLike | None = None,
    mass_range: tuple[float, float] | None = None,
    resolution: float | None = None,
    units: str = "ppm",
    guess_max: int = 1000,
    tolerance: float | None = None,
) -> MSImagingExperiment:
    """Bin ragged spectra onto a shared m/z axis and return an experiment.

    If all spectra already share exactly the same axis, it is reused. For
    varying axes, supply ``mz`` or both ``mass_range`` and ``resolution``.
    Nearest-axis bins outside ``tolerance`` are omitted; default tolerance is
    half the requested resolution (or half a local axis interval).
    """
    if units not in {"mz", "ppm"}:
        raise ValueError("units must be 'mz' or 'ppm'")
    if guess_max < 1:
        raise ValueError("guess_max must be positive")
    if mz is not None:
        axis = np.asarray(mz, dtype=float)
        if axis.ndim != 1 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError("mz must be a finite, strictly increasing vector")
    elif len(obj) == 0:
        axis = np.array([], dtype=float)
    else:
        first_axis = np.asarray(obj.mz[0], dtype=float)
        shared = all(
            np.array_equal(first_axis, np.asarray(values, dtype=float))
            for values in obj.mz[1:]
        )
        if shared:
            axis = first_axis.copy()
        else:
            if resolution is None:
                raise ValueError(
                    "resolution is required when spectra do not share an m/z axis"
                )
            inferred_range = mass_range or _sampled_mass_range(obj, guess_max)
            axis = _make_mz_axis(inferred_range, resolution, units)
    if mass_range is not None:
        lower, upper = mass_range
        if not np.isfinite([lower, upper]).all() or lower <= 0 or upper < lower:
            raise ValueError(
                "mass_range must contain finite positive increasing bounds"
            )
        axis = axis[(axis >= lower) & (axis <= upper)]
    if resolution is None and np.size(axis) > 1:
        spacing = np.median(np.diff(axis))
        default_tolerance = (
            float(spacing / np.median(axis) * 1e6 / 2)
            if units == "ppm"
            else float(spacing / 2)
        )
    elif resolution is not None:
        default_tolerance = float(resolution / 2)
    else:
        default_tolerance = 0.0
    match_tolerance = default_tolerance if tolerance is None else float(tolerance)
    if match_tolerance < 0 or not np.isfinite(match_tolerance):
        raise ValueError("tolerance must be a finite, nonnegative number")

    rows: list[NDArray[np.int64]] = []
    columns: list[NDArray[np.int64]] = []
    values: list[NDArray[np.float64]] = []
    for pixel_index, (masses, signal) in enumerate(zip(obj.mz, obj.intensity)):
        masses_array = np.asarray(masses, dtype=float)
        signal_array = np.asarray(signal, dtype=float)
        if masses_array.size == 0 or axis.size == 0:
            continue
        right = np.searchsorted(axis, masses_array).clip(0, len(axis) - 1)
        left = np.maximum(right - 1, 0)
        choose_left = np.abs(masses_array - axis[left]) <= np.abs(
            masses_array - axis[right]
        )
        feature_indices = np.where(choose_left, left, right)
        errors = np.abs(masses_array - axis[feature_indices])
        if units == "ppm":
            within_tolerance = (errors / masses_array) * 1e6 <= match_tolerance
        else:
            within_tolerance = errors <= match_tolerance
        keep = within_tolerance
        if np.any(keep):
            rows.append(feature_indices[keep].astype(np.int64))
            columns.append(np.full(np.count_nonzero(keep), pixel_index, dtype=np.int64))
            values.append(signal_array[keep])
    if values:
        matrix = sparse.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(len(axis), len(obj)),
        ).tocsr()
    else:
        matrix = sparse.csr_matrix((len(axis), len(obj)), dtype=float)
    return MSImagingExperiment(
        {"intensity": matrix},
        feature_data=MassDataFrame(axis),
        pixel_data=obj.pixel_data,
        experiment_data=obj.experiment_data,
        centroided=obj.centroided,
        metadata=obj.metadata,
    )


def convert_experiment_to_arrays(obj: MSImagingExperiment) -> MSImagingArrays:
    """Return nonzero per-pixel spectra from a shared-domain experiment."""
    matrix = obj.intensity
    masses: list[NDArray[np.float64]] = []
    signals: list[NDArray[np.float64]] = []
    for pixel_index in range(obj.shape[1]):
        column = (
            np.asarray(cast(Any, matrix).getcol(pixel_index).toarray()).ravel()
            if sparse.issparse(matrix)
            else np.asarray(matrix)[:, pixel_index]
        )
        keep = column != 0
        masses.append(obj.mz[keep].copy())
        signals.append(np.asarray(column[keep], dtype=float).copy())
    return MSImagingArrays(
        mz=masses,
        intensity=signals,
        pixel_data=obj.pixel_data,
        experiment_data=obj.experiment_data,
        centroided=obj.centroided,
        continuous=False,
        processing=obj.processing,
        metadata=obj.metadata,
    )


def write_imzml(
    obj: MSImagingArrays | MSImagingExperiment,
    file: str | Path,
    *,
    bundle: bool = True,
    verbose: bool = False,
) -> Path:
    """Write a dataset and its binary sidecar as an imzML pair.

    With ``bundle=True``, ``file`` names a directory and output XML uses that
    directory's name. With ``bundle=False``, ``file`` is the output XML path.
    """
    path = Path(file).expanduser()
    if bundle:
        bundle_dir = path.with_suffix("") if path.suffix.lower() == ".imzml" else path
        bundle_dir.mkdir(parents=True, exist_ok=True)
        output_path = bundle_dir / f"{bundle_dir.name}.imzML"
    else:
        output_path = (
            path if path.suffix.lower() == ".imzml" else path.with_suffix(".imzML")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = output_path.with_suffix("")
    if verbose:
        _LOGGER.info("Writing imzML file %s", output_path)

    coordinates = obj.coord[["x", "y"]].to_numpy(dtype=float)
    if "z" in obj.pixel_data:
        coordinates = obj.coord[["x", "y", "z"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or np.any(coordinates <= 0):
        raise ValueError("imzML coordinates must be finite and positive (1-based)")
    rounded = np.rint(coordinates)
    if not np.array_equal(coordinates, rounded):
        raise ValueError("imzML coordinates must be integer pixel positions")
    if obj.run.categories.size > 1:
        raise NotImplementedError("writing multiple imzML runs is not implemented yet")
    writer_mode = "continuous" if isinstance(obj, MSImagingExperiment) else "processed"
    spec_type = "centroid" if obj.centroided is True else "profile"
    with ImzMLWriter(
        str(prefix),
        mode=writer_mode,
        spec_type=spec_type,
    ) as writer:
        if isinstance(obj, MSImagingExperiment):
            matrix = obj.intensity
            for pixel_index, coord in enumerate(rounded.astype(int)):
                signal = (
                    np.asarray(cast(Any, matrix).getcol(pixel_index).toarray()).ravel()
                    if sparse.issparse(matrix)
                    else np.asarray(matrix)[:, pixel_index]
                )
                writer.addSpectrum(obj.mz, signal, tuple(coord.tolist()))
        else:
            for masses, signal, coord in zip(
                obj.mz, obj.intensity, rounded.astype(int)
            ):
                writer.addSpectrum(masses, signal, tuple(coord.tolist()))
    return output_path
