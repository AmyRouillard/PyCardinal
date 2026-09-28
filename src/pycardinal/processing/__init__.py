"""Spectral preprocessing operations."""

from pycardinal.processing.baseline import reduce_baseline
from pycardinal.processing.normalize import normalize
from pycardinal.processing.peaks import (
    bin_spectra,
    estimate_domain,
    estimate_reference_mz,
    estimate_reference_peaks,
    peak_align,
    peak_pick,
    peak_process,
)
from pycardinal.processing.recalibrate import recalibrate
from pycardinal.processing.smooth import smooth

__all__ = [
    "bin_spectra",
    "estimate_domain",
    "estimate_reference_mz",
    "estimate_reference_peaks",
    "normalize",
    "peak_align",
    "peak_pick",
    "peak_process",
    "recalibrate",
    "reduce_baseline",
    "smooth",
]
