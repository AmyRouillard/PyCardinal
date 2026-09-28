"""Pandas-backed metadata frames for pixels and mass features."""

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


class XDFrame(pd.DataFrame):
    """DataFrame base that records columns identifying domain keys.

    ``key_columns`` is used instead of ``keys`` to preserve pandas'
    ``DataFrame.keys()`` method.
    """

    _metadata = ["key_columns"]
    key_columns: dict[str, str]

    def __init__(
        self,
        data: Any = None,
        key_columns: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        cast(Any, pd.DataFrame).__init__(self, data=data, **kwargs)
        self.key_columns = dict(key_columns or {})

    @property
    def _constructor(self) -> type[pd.DataFrame]:
        return pd.DataFrame


class PositionDataFrame(XDFrame):
    """Pixel metadata with numeric ``x``/``y`` coordinates and a ``run``.

    Coordinates can be supplied as a DataFrame, a mapping, or an ``N x 2``
    (or ``N x 3``) array. If omitted, ``run`` defaults to a single run.
    """

    def __init__(
        self,
        coord: ArrayLike | Mapping[str, ArrayLike] | pd.DataFrame | None = None,
        run: ArrayLike | None = None,
        **columns: ArrayLike,
    ) -> None:
        if isinstance(coord, pd.DataFrame):
            frame = pd.DataFrame(coord).copy()
        elif isinstance(coord, Mapping):
            frame = pd.DataFrame(coord)
        elif coord is None:
            frame = pd.DataFrame()
        else:
            coord_values = np.asarray(coord)
            if coord_values.ndim != 2 or coord_values.shape[1] not in (2, 3):
                raise ValueError("coord must have two or three columns")
            frame = pd.DataFrame(
                coord_values,
                columns=["x", "y", "z"][: coord_values.shape[1]],
            )

        for name, column_values in columns.items():
            frame[name] = cast(Any, column_values)
        if run is not None:
            frame["run"] = cast(Any, run)
        elif "run" not in frame:
            frame["run"] = ["run1"] * len(frame)

        missing = {"x", "y"}.difference(frame.columns)
        if missing:
            raise ValueError(
                f"pixel coordinates are missing columns: {sorted(missing)}"
            )
        coordinate_names = [name for name in ("x", "y", "z") if name in frame]
        try:
            coordinate_values = frame[coordinate_names].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("pixel coordinates must be numeric") from exc
        if not np.isfinite(coordinate_values).all():
            raise ValueError("pixel coordinates must be finite")

        super().__init__(
            frame,
            key_columns={"coord": ",".join(coordinate_names), "run": "run"},
        )

    @property
    def coord(self) -> pd.DataFrame:
        """Coordinate columns in their original x/y/z order."""
        names = [name for name in ("x", "y", "z") if name in self.columns]
        return pd.DataFrame(self.loc[:, names]).copy()

    @property
    def run(self) -> pd.Categorical:
        """Categorical run labels, one per pixel."""
        return pd.Categorical(self["run"])


class MassDataFrame(XDFrame):
    """Feature metadata with a finite, nondecreasing ``mz`` column."""

    def __init__(
        self,
        mz: ArrayLike | Mapping[str, ArrayLike] | pd.DataFrame | None = None,
        **columns: ArrayLike,
    ) -> None:
        if isinstance(mz, pd.DataFrame):
            frame = pd.DataFrame(mz).copy()
        elif isinstance(mz, Mapping):
            frame = pd.DataFrame(mz)
        elif mz is None:
            frame = pd.DataFrame()
        else:
            mz_values = np.asarray(mz)
            if mz_values.ndim != 1:
                raise ValueError("mz must be one-dimensional")
            frame = pd.DataFrame({"mz": mz_values})

        for name, column_values in columns.items():
            frame[name] = cast(Any, column_values)
        if "mz" not in frame:
            raise ValueError("feature metadata must include an 'mz' column")
        try:
            masses = pd.to_numeric(frame["mz"], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("mz values must be numeric") from exc
        if not np.isfinite(masses).all():
            raise ValueError("mz values must be finite")
        if np.any(np.diff(masses) < 0):
            raise ValueError("mz values must be sorted in nondecreasing order")
        frame["mz"] = cast(Any, masses)
        super().__init__(frame, key_columns={"mz": "mz"})

    @property
    def mz(self) -> NDArray[np.float64]:
        """Sorted m/z values as a NumPy array."""
        return self["mz"].to_numpy(dtype=float)


XDataFrame = XDFrame
