"""Compare fixed-seed Python summaries with a JSON artifact produced by R.

The R command is intentionally external because Cardinal/R may not be installed
in the Python environment. The R side should write a JSON object containing the
same numeric keys as the Python artifact, for example ``feature_mean``,
``pixel_tic``, and ``pca_sdev``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pycardinal import (  # type: ignore[import-untyped]
    PCA,
    MassDataFrame,
    MSImagingExperiment,
    PositionDataFrame,
)
from pycardinal.summarize import (  # type: ignore[import-untyped]
    summarize_features,
    summarize_pixels,
)


def python_reference() -> dict[str, list[float]]:
    matrix = np.array(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [4.0, 3.0, 2.0, 1.0]],
        dtype=float,
    )
    experiment = MSImagingExperiment(
        matrix,
        feature_data=MassDataFrame([100.0, 200.0, 300.0]),
        pixel_data=PositionDataFrame(
            {"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]}
        ),
    )
    feature_summary = summarize_features(experiment)
    pixel_summary = summarize_pixels(experiment)
    pca = PCA(experiment, ncomp=2)
    return {
        "feature_mean": feature_summary.feature_data["mean"].tolist(),
        "pixel_tic": pixel_summary.pixel_data["tic"].tolist(),
        "pca_sdev": pca.sdev.tolist(),
    }


def _compare(expected: Any, actual: Any, tolerance: float, path: str = "root") -> None:
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: length differs")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, tolerance, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not np.isclose(expected, actual, rtol=tolerance, atol=tolerance):
            raise AssertionError(f"{path}: {expected!r} != {actual!r}")
        return
    if expected != actual:
        raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-output", type=Path, required=True)
    parser.add_argument("--r-output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    reference = python_reference()
    args.python_output.write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    if args.r_output is None:
        print(f"Wrote Python reference to {args.python_output}")
        return
    r_values = json.loads(args.r_output.read_text(encoding="utf-8"))
    _compare(reference, r_values, args.tolerance)
    print(f"Comparison passed within tolerance {args.tolerance:g}")


if __name__ == "__main__":
    main()
