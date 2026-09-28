"""Named, shape-compatible storage for spectral data arrays."""

from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any


def _shape(value: Any) -> tuple[int, ...]:
    value_shape = getattr(value, "shape", None)
    if value_shape is not None:
        return tuple(int(size) for size in value_shape)
    try:
        return (len(value),)
    except TypeError as exc:
        raise TypeError("spectra arrays must be sized array-like values") from exc


class SpectraArrays(MutableMapping[str, Any]):
    """Store named arrays with identical shapes.

    Ragged spectra may be stored as sequences: their shared outer dimension is
    checked here, while each spectrum's internal length may differ.
    """

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        if data is not None:
            for name, value in data.items():
                self[name] = value

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def __setitem__(self, name: str, value: Any) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("array names must be non-empty strings")
        new_shape = _shape(value)
        for other_name, other_value in self._data.items():
            if other_name != name and _shape(other_value) != new_shape:
                raise ValueError(
                    f"array {name!r} has shape {new_shape}, but {other_name!r} "
                    f"has shape {_shape(other_value)}"
                )
        self._data[name] = value

    def __delitem__(self, name: str) -> None:
        del self._data[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def names(self) -> list[str]:
        """Names of the contained arrays in insertion order."""
        return list(self._data)

    def copy(self) -> "SpectraArrays":
        """Return a new container sharing the stored array values."""
        return SpectraArrays(self._data)
