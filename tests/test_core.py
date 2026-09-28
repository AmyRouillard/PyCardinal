import numpy as np
import pytest
from scipy import sparse

from pycardinal import (
    MassDataFrame,
    MSImagingArrays,
    MSImagingExperiment,
    PositionDataFrame,
    SpectraArrays,
    add_processing,
    process,
    reset,
)


def make_pixels(count: int) -> PositionDataFrame:
    return PositionDataFrame(
        {"x": np.arange(count), "y": np.zeros(count), "run": ["r1"] * count}
    )


def multiply_signal(signal: np.ndarray, mz: np.ndarray, factor: float) -> np.ndarray:
    return signal * factor


def add_signal(signal: np.ndarray, mz: np.ndarray, value: float) -> np.ndarray:
    return signal + value


def mutate_signal(signal: np.ndarray, mz: np.ndarray) -> np.ndarray:
    signal += 1
    return signal


def shift_mass_axis(
    signal: np.ndarray, mz: np.ndarray, shift: float
) -> tuple[np.ndarray, np.ndarray]:
    return mz + shift, signal


def keep_first_two(signal: np.ndarray, mz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return mz[:2], signal[:2]


def test_spectra_arrays_require_compatible_shapes() -> None:
    arrays = SpectraArrays({"intensity": np.ones((3, 2))})

    with pytest.raises(ValueError, match="shape"):
        arrays["mask"] = np.ones((3, 1))


def test_metadata_validates_coordinates_and_sorted_mz() -> None:
    pixels = make_pixels(2)
    masses = MassDataFrame([100.0, 101.0], label=["a", "b"])

    assert pixels.coord.columns.tolist() == ["x", "y"]
    assert pixels.run.tolist() == ["r1", "r1"]
    assert masses.mz.tolist() == [100.0, 101.0]

    with pytest.raises(ValueError, match="missing columns"):
        PositionDataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="sorted"):
        MassDataFrame([101.0, 100.0])


def test_experiment_shape_orientation_and_dense_subsetting() -> None:
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    experiment = MSImagingExperiment(
        matrix,
        feature_data=MassDataFrame([100.0, 101.0, 102.0]),
        pixel_data=make_pixels(4),
        centroided=True,
    )

    selected = experiment[1:, [0, 2]]

    assert experiment.shape == (3, 4)
    assert len(experiment) == 4
    assert experiment.is_centroided()
    assert selected.shape == (2, 2)
    np.testing.assert_array_equal(selected.intensity, matrix[1:, [0, 2]])
    np.testing.assert_array_equal(selected.mz, [101.0, 102.0])
    assert selected.coord["x"].tolist() == [0, 2]


def test_experiment_supports_sparse_subsetting() -> None:
    matrix = sparse.csr_matrix(np.arange(12, dtype=float).reshape(3, 4))
    experiment = MSImagingExperiment(
        {"intensity": matrix},
        feature_data=MassDataFrame([100.0, 101.0, 102.0]),
        pixel_data=make_pixels(4),
    )

    selected = experiment[[0, 2], [1, 3]]

    assert sparse.issparse(selected.intensity)
    np.testing.assert_array_equal(
        selected.intensity.toarray(), [[1.0, 3.0], [9.0, 11.0]]
    )


def test_ragged_spectra_arrays_validate_each_spectrum() -> None:
    spectra = MSImagingArrays(
        mz=[np.array([100.0, 101.0]), np.array([200.0])],
        intensity=[np.array([1.0, 2.0]), np.array([3.0])],
        pixel_data=make_pixels(2),
    )

    assert len(spectra) == 2
    assert [len(spectrum) for spectrum in spectra.mz] == [2, 1]

    with pytest.raises(ValueError, match="lengths differ"):
        MSImagingArrays(
            mz=[np.array([100.0, 101.0])],
            intensity=[np.array([1.0])],
            pixel_data=make_pixels(1),
        )


def test_deferred_processing_order_and_copy_semantics() -> None:
    original_matrix = np.arange(6, dtype=float).reshape(2, 3)
    original = MSImagingExperiment(
        original_matrix,
        feature_data=MassDataFrame([100.0, 101.0]),
        pixel_data=make_pixels(3),
    )
    queued = add_processing(original, multiply_signal, "multiply", factor=2)
    queued = add_processing(queued, add_signal, "add", value=3)

    result = process(queued)

    np.testing.assert_array_equal(result.intensity, original_matrix * 2 + 3)
    np.testing.assert_array_equal(original.intensity, original_matrix)
    assert original.processing == []
    assert len(queued.processing) == 2
    assert result.processing == []


def test_processing_can_update_a_shared_mass_axis() -> None:
    experiment = MSImagingExperiment(
        np.ones((2, 2)),
        feature_data=MassDataFrame([100.0, 101.0]),
        pixel_data=make_pixels(2),
    )

    result = process(add_processing(experiment, shift_mass_axis, "shift", shift=0.1))

    np.testing.assert_allclose(result.mz, [100.1, 101.1])


def test_processing_can_change_shared_feature_count() -> None:
    experiment = MSImagingExperiment(
        np.arange(6, dtype=float).reshape(3, 2),
        feature_data=MassDataFrame([100.0, 101.0, 102.0]),
        pixel_data=make_pixels(2),
    )

    result = process(add_processing(experiment, keep_first_two, "crop"))

    assert result.shape == (2, 2)
    assert len(result.feature_data) == 2
    np.testing.assert_array_equal(result.mz, [100.0, 101.0])


def test_reset_discards_queued_steps_without_mutating_input() -> None:
    experiment = MSImagingExperiment(
        np.ones((1, 2)),
        feature_data=MassDataFrame([100.0]),
        pixel_data=make_pixels(2),
    )
    queued = add_processing(experiment, add_signal, "add", value=5)

    cleared = reset(queued)

    assert len(queued.processing) == 1
    assert cleared.processing == []
    np.testing.assert_array_equal(cleared.intensity, experiment.intensity)


def test_in_place_processing_does_not_mutate_original_data() -> None:
    experiment = MSImagingExperiment(
        np.ones((1, 2)),
        feature_data=MassDataFrame([100.0]),
        pixel_data=make_pixels(2),
    )

    result = process(add_processing(experiment, mutate_signal, "mutate"))

    np.testing.assert_array_equal(experiment.intensity, [[1.0, 1.0]])
    np.testing.assert_array_equal(result.intensity, [[2.0, 2.0]])
