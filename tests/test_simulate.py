import numpy as np
import pandas as pd

from pycardinal import (
    MassDataFrame,
    MSImagingArrays,
    MSImagingExperiment,
    PositionDataFrame,
    add_shape,
    preset_image_def,
    simulate_image,
    simulate_spectra,
)


def test_simulate_spectra_is_reproducible_and_returns_peak_table() -> None:
    first = simulate_spectra(n=2, npeaks=8, random_state=42)
    second = simulate_spectra(n=2, npeaks=8, random_state=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["mz"].is_monotonic_increasing
    assert {"intensity_1", "intensity_2"}.issubset(first.columns)


def test_simulate_centroided_spectra_use_the_peak_axis() -> None:
    result = simulate_spectra(
        n=3,
        mz=[100.0, 200.0, 300.0],
        intensity=[1.0, 2.0, 3.0],
        centroided=True,
        random_state=4,
    )

    assert result["mz"].tolist() == [100.0, 200.0, 300.0]
    assert result.shape == (3, 4)


def test_add_shape_creates_circle_and_square_masks() -> None:
    pixels = PositionDataFrame({"x": [1, 2, 3], "y": [1, 2, 1], "run": ["r1"] * 3})

    circle = add_shape(pixels, center={"x": 1, "y": 1}, size=1, shape="circle")
    square = add_shape(pixels, center=[1, 1], size=1, shape="square", name="roi")

    assert circle["circle"].tolist() == [True, False, False]
    assert square["roi"].tolist() == [True, True, False]
    assert "circle" not in pixels


def test_preset_design_has_matching_feature_and_pixel_regions() -> None:
    design = preset_image_def(
        preset=2,
        nrun=2,
        npeaks=9,
        dim=(5, 6),
        random_state=13,
    )
    pixels = design["pixel_data"]
    features = design["feature_data"]

    assert len(pixels) == 60
    assert len(features) == 9
    assert {"circle", "square"}.issubset(pixels.columns)
    assert {"circle", "square"}.issubset(features.columns)
    assert pixels["run"].nunique() == 2


def test_all_nine_presets_generate_compatible_images() -> None:
    for preset in range(1, 10):
        dimensions = (4, 5, 2) if preset == 9 else (4, 5)
        design = preset_image_def(
            preset=preset,
            nrun=2,
            npeaks=6,
            dim=dimensions,
            jitter=False,
            random_state=100 + preset,
        )
        image = simulate_image(
            preset=preset,
            nrun=2,
            npeaks=6,
            dim=dimensions,
            jitter=False,
            by=50_000,
            random_state=100 + preset,
        )

        assert len(image) == len(design["pixel_data"])
        assert image.shape[0] == len(image.mz)
        assert image.shape[1] == len(design["pixel_data"])


def test_preset_nine_does_not_duplicate_its_z_layers() -> None:
    design = preset_image_def(
        preset=9,
        nrun=2,
        npeaks=6,
        dim=(4, 5, 2),
        jitter=False,
        random_state=9,
    )

    assert len(design["pixel_data"]) == 4 * 5 * 2


def test_custom_design_supports_seeded_spatial_noise_and_mass_range() -> None:
    pixels = PositionDataFrame(
        {"x": [1, 2, 3], "y": [1, 1, 1], "region": [True, True, False]}
    )
    features = MassDataFrame([100.0, 200.0], region=[4.0, 2.0])

    first = simulate_image(
        pixel_data=pixels,
        feature_data=features,
        by=50_000,
        spcorr=0.5,
        sar=True,
        random_state=31,
    )
    second = simulate_image(
        pixel_data=pixels,
        feature_data=features,
        by=50_000,
        spcorr=0.5,
        sar=True,
        random_state=31,
    )

    np.testing.assert_array_equal(first.intensity, second.intensity)
    np.testing.assert_array_equal(
        first.metadata["design"]["feature_data"].mz,
        features.mz,
    )


def test_preset_mass_axis_maps_to_requested_range() -> None:
    image = simulate_image(
        preset=1,
        npeaks=6,
        dim=(3, 3),
        from_=500,
        to=1000,
        by=50_000,
        jitter=False,
        random_state=18,
    )
    generated_mz = image.metadata["design"]["feature_data"].mz

    assert generated_mz.min() >= 550
    assert generated_mz.max() <= 950


def test_simulate_image_returns_seeded_continuous_experiment() -> None:
    first = simulate_image(
        preset=1,
        npeaks=6,
        dim=(4, 3),
        by=20_000,
        random_state=22,
    )
    second = simulate_image(
        preset=1,
        npeaks=6,
        dim=(4, 3),
        by=20_000,
        random_state=22,
    )

    assert isinstance(first, MSImagingExperiment)
    assert first.shape == (len(first.mz), 12)
    np.testing.assert_array_equal(first.intensity, second.intensity)
    assert first.metadata["design"]["pixel_data"].shape == (12, 4)


def test_simulate_image_can_create_processed_centroided_spectra() -> None:
    image = simulate_image(
        preset=1,
        npeaks=5,
        dim=(3, 2),
        centroided=True,
        continuous=False,
        random_state=6,
    )

    assert isinstance(image, MSImagingArrays)
    assert len(image) == 6
    assert all(len(masses) == 5 for masses in image.mz)
    assert all(len(signal) == 5 for signal in image.intensity)
