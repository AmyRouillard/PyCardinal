"""Mass spectrometry imaging file input and output."""

from pathlib import Path
from typing import Any

from pycardinal.core import MSImagingArrays, MSImagingExperiment
from pycardinal.core.imaging_data import SpectralImagingData
from pycardinal.io.imzml import (
    convert_arrays_to_experiment,
    convert_experiment_to_arrays,
    read_imzml,
    write_imzml,
)


def read_msi_data(file: str | Path, **kwargs: Any) -> Any:
    """Read a supported mass-spectrometry imaging file by extension."""
    path = Path(file)
    suffix = path.suffix.lower()
    if suffix in {".imzml", ".ibd"}:
        if suffix == ".ibd":
            candidates = [path.with_suffix(".imzML"), path.with_suffix(".imzml")]
            path = next(
                (candidate for candidate in candidates if candidate.is_file()),
                candidates[0],
            )
        return read_imzml(path, **kwargs)
    if suffix in {".img", ".hdr", ".t2m"}:
        raise NotImplementedError("Analyze 7.5 I/O is not implemented yet")
    raise ValueError(f"cannot recognize mass-imaging file extension: {suffix!r}")


def write_msi_data(
    obj: MSImagingArrays | MSImagingExperiment,
    file: str | Path,
    **kwargs: Any,
) -> Path:
    """Write a supported mass-spectrometry imaging file by extension."""
    path = Path(file)
    suffix = path.suffix.lower()
    if suffix in {".imzml", ".ibd", ""}:
        if suffix == ".ibd":
            path = path.with_suffix(".imzML")
        return write_imzml(obj, path, **kwargs)
    if suffix in {".img", ".hdr", ".t2m"}:
        raise NotImplementedError("Analyze 7.5 I/O is not implemented yet")
    raise ValueError(f"cannot recognize mass-imaging file extension: {suffix!r}")


def read_analyze(file: str | Path, **kwargs: Any) -> SpectralImagingData:
    """Read Analyze 7.5 data (not implemented yet)."""
    del file, kwargs
    raise NotImplementedError("Analyze 7.5 I/O is not implemented yet")


def write_analyze(
    obj: MSImagingArrays | MSImagingExperiment,
    file: str | Path,
    **kwargs: Any,
) -> None:
    """Write Analyze 7.5 data (not implemented yet)."""
    del obj, file, kwargs
    raise NotImplementedError("Analyze 7.5 I/O is not implemented yet")


__all__ = [
    "convert_arrays_to_experiment",
    "convert_experiment_to_arrays",
    "read_analyze",
    "read_imzml",
    "read_msi_data",
    "write_analyze",
    "write_imzml",
    "write_msi_data",
]
