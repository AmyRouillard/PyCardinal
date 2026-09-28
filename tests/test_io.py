from pathlib import Path

import numpy as np
import pytest

from pycardinal import (
    MassDataFrame,
    MSImagingArrays,
    MSImagingExperiment,
    PositionDataFrame,
    convert_arrays_to_experiment,
    convert_experiment_to_arrays,
    read_imzml,
    read_msi_data,
    simulate_image,
    write_imzml,
    write_msi_data,
)


def make_pixels(coords: list[tuple[int, int]]) -> PositionDataFrame:
    return PositionDataFrame({"x": [x for x, _ in coords], "y": [y for _, y in coords]})


@pytest.fixture
def simulated_imzml(tmp_path: Path) -> tuple[Path, MSImagingExperiment]:
    source = simulate_image(
        preset=1,
        npeaks=8,
        dim=(3, 4),
        by=10_000,
        jitter=False,
        random_state=17,
    )
    assert isinstance(source, MSImagingExperiment)
    output = write_imzml(source, tmp_path / "simulated.imzML", bundle=False)
    return output, source


def test_simulated_imzml_reads_without_external_test_data(
    simulated_imzml: tuple[Path, MSImagingExperiment],
) -> None:
    path, source = simulated_imzml
    dataset = read_imzml(path)

    assert isinstance(dataset, MSImagingExperiment)
    assert dataset.experiment_data["representation"] == "continuous"
    assert dataset.shape == source.shape
    np.testing.assert_allclose(dataset.mz, source.mz)
    np.testing.assert_allclose(dataset.intensity, source.intensity, atol=1e-6)


def test_parse_only_returns_metadata_without_spectra(
    simulated_imzml: tuple[Path, MSImagingExperiment],
) -> None:
    path, source = simulated_imzml
    parsed = read_imzml(path, parse_only=True)

    assert parsed["representation"] == "continuous"
    assert parsed["centroided"] is False
    assert len(parsed["coordinates"]) == len(source)


def test_processed_spectra_convert_to_shared_mz_axis() -> None:
    arrays = MSImagingArrays(
        mz=[np.array([100.0, 101.0]), np.array([100.1, 101.1])],
        intensity=[np.array([2.0, 3.0]), np.array([4.0, 5.0])],
        pixel_data=make_pixels([(1, 1), (2, 1)]),
    )

    experiment = convert_arrays_to_experiment(
        arrays,
        mz=[100.0, 101.0],
        units="mz",
        tolerance=0.2,
    )

    assert isinstance(experiment, MSImagingExperiment)
    assert experiment.shape == (2, 2)
    np.testing.assert_allclose(experiment.intensity.toarray(), [[2.0, 4.0], [3.0, 5.0]])

    recovered = convert_experiment_to_arrays(experiment)
    assert [len(values) for values in recovered.mz] == [2, 2]
    np.testing.assert_allclose(recovered.intensity[1], [4.0, 5.0])


def test_processed_imzml_write_read_round_trip(tmp_path: Path) -> None:
    source = MSImagingArrays(
        mz=[np.array([100.0, 101.0]), np.array([200.0])],
        intensity=[np.array([2.0, 3.0]), np.array([5.0])],
        pixel_data=make_pixels([(1, 1), (2, 1)]),
        centroided=True,
    )

    output = write_imzml(source, tmp_path / "processed.imzML", bundle=False)
    loaded = read_imzml(output)

    assert isinstance(loaded, MSImagingArrays)
    np.testing.assert_allclose(loaded.mz[0], source.mz[0])
    np.testing.assert_allclose(loaded.intensity[0], source.intensity[0])
    np.testing.assert_allclose(loaded.mz[1], source.mz[1])
    np.testing.assert_allclose(loaded.intensity[1], source.intensity[1])


def test_continuous_imzml_write_read_round_trip_and_ibd_dispatch(
    tmp_path: Path,
) -> None:
    source = MSImagingExperiment(
        np.array([[2.0, 4.0], [3.0, 5.0]]),
        feature_data=MassDataFrame([100.0, 101.0]),
        pixel_data=make_pixels([(1, 1), (2, 1)]),
        centroided=False,
    )

    output = write_imzml(source, tmp_path / "continuous.imzML", bundle=False)
    loaded = read_msi_data(output.with_suffix(".ibd"))

    assert isinstance(loaded, MSImagingExperiment)
    assert loaded.experiment_data["representation"] == "continuous"
    np.testing.assert_allclose(loaded.mz, source.mz)
    np.testing.assert_allclose(loaded.intensity, source.intensity)


def test_write_dispatch_accepts_ibd_sidecar_path(tmp_path: Path) -> None:
    source = MSImagingArrays(
        mz=[np.array([100.0])],
        intensity=[np.array([2.0])],
        pixel_data=make_pixels([(1, 1)]),
        centroided=True,
    )

    output = write_msi_data(
        source,
        tmp_path / "sidecar.ibd",
        bundle=False,
    )

    assert output.name == "sidecar.imzML"
    assert output.with_suffix(".ibd").is_file()


def test_processed_to_experiment_requires_resolution_or_shared_axis() -> None:
    arrays = MSImagingArrays(
        mz=[np.array([100.0]), np.array([101.0])],
        intensity=[np.array([2.0]), np.array([3.0])],
        pixel_data=make_pixels([(1, 1), (2, 1)]),
    )

    with pytest.raises(ValueError, match="resolution is required"):
        convert_arrays_to_experiment(arrays)
