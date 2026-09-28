import numpy as np
import pytest
from scipy import sparse

from pycardinal import (
    MassDataFrame,
    MSImagingArrays,
    MSImagingExperiment,
    PositionDataFrame,
    colocalized,
    features,
    pixels,
    slice_image,
    subset,
    subset_features,
    subset_pixels,
)


def make_experiment(sparse_matrix: bool = False) -> MSImagingExperiment:
    matrix = np.arange(12, dtype=float).reshape(4, 3)
    if sparse_matrix:
        matrix = sparse.csr_matrix(matrix)
    spectra_data = {"intensity": matrix} if sparse_matrix else matrix
    return MSImagingExperiment(
        spectra_data,
        feature_data=MassDataFrame(
            [100.0, 200.0, 300.0, 400.0], quality=[0.2, 0.8, 0.9, 0.5]
        ),
        pixel_data=PositionDataFrame(
            {
                "x": [0, 1, 2],
                "y": [0, 0, 0],
                "region": ["a", "b", "b"],
                "quality": [0.2, 0.8, 0.9],
                "run": ["r1", "r1", "r2"],
            }
        ),
    )


def test_features_combines_query_and_keyword_conditions() -> None:
    experiment = make_experiment()
    selected = features(experiment, query="mz >= 200", quality__gt=0.5, mz__lt=400)

    np.testing.assert_array_equal(selected, [1, 2])


def test_feature_conditions_support_values_and_callables() -> None:
    experiment = make_experiment()
    selected = features(
        experiment,
        mz__in=[100.0, 400.0],
        quality=lambda values: values < 0.5,
    )

    np.testing.assert_array_equal(selected, [0])


def test_pixels_and_subset_preserve_metadata_and_dimensions() -> None:
    experiment = make_experiment()
    selected = pixels(experiment, query="x >= 1", run="r1")
    result = subset_pixels(experiment, query="x >= 1", run="r1")

    np.testing.assert_array_equal(selected, [1])
    assert result.shape == (4, 1)
    assert result.pixel_data["region"].tolist() == ["b"]
    np.testing.assert_array_equal(result.intensity, [[1.0], [4.0], [7.0], [10.0]])


def test_subset_features_and_index_subset_keep_feature_metadata() -> None:
    experiment = make_experiment()
    selected = subset_features(experiment, mz__ge=200, mz__le=300)
    indexed = subset(experiment, select=[1, 3], subset=[2, 0])

    assert selected.shape == (2, 3)
    np.testing.assert_array_equal(selected.mz, [200.0, 300.0])
    assert isinstance(indexed.feature_data, MassDataFrame)
    np.testing.assert_array_equal(indexed.mz, [200.0, 400.0])
    np.testing.assert_array_equal(indexed.intensity, [[5.0, 3.0], [11.0, 9.0]])


def test_colocalized_ranks_highly_correlated_features() -> None:
    experiment = MSImagingExperiment(
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
                [8.0, 6.0, 4.0, 2.0],
            ],
            dtype=float,
        ),
        feature_data=MassDataFrame([100.0, 200.0, 300.0]),
        pixel_data=PositionDataFrame(
            {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 0.0, 0.0, 0.0]}
        ),
    )

    result = colocalized(experiment, mz=100.0, sort_by="cor")

    assert result.iloc[0]["i"] == 0
    assert result.iloc[0]["cor"] == pytest.approx(1.0)
    assert result.iloc[0]["mz"] == 100.0


def test_sparse_subsetting_preserves_sparse_storage() -> None:
    result = subset(make_experiment(sparse_matrix=True), select=[0, 2], subset=[1])

    assert sparse.issparse(result.intensity)
    np.testing.assert_array_equal(result.intensity.toarray(), [[1.0], [7.0]])


def test_ragged_pixel_subsetting_keeps_spectra_and_metadata() -> None:
    experiment = make_experiment()
    arrays = MSImagingArrays(
        mz=[np.array([100.0]), np.array([200.0, 201.0]), np.array([300.0])],
        intensity=[np.array([1.0]), np.array([2.0, 3.0]), np.array([4.0])],
        pixel_data=experiment.pixel_data,
        continuous=True,
        metadata={"source": "test"},
    )

    result = subset_pixels(arrays, region="b")

    assert isinstance(result, MSImagingArrays)
    assert len(result) == 2
    np.testing.assert_array_equal(result.mz[0], [200.0, 201.0])
    np.testing.assert_array_equal(result.intensity[1], [4.0])
    assert result.pixel_data["x"].tolist() == [1, 2]
    assert result.continuous
    assert result.metadata == {"source": "test"}


def test_selection_validation_and_ragged_feature_selection() -> None:
    experiment = make_experiment()
    arrays = MSImagingArrays(
        mz=[np.array([100.0])],
        intensity=[np.array([1.0])],
        pixel_data=experiment.pixel_data.iloc[:1],
    )

    with pytest.raises(KeyError, match="no column"):
        features(experiment, absent__gt=1)
    with pytest.raises(ValueError, match="one boolean value"):
        pixels(experiment, x=lambda values: [True])
    with pytest.raises(TypeError, match="shared-domain"):
        subset(arrays, select=[0])


def test_slice_image_rasterizes_coordinates_and_fills_missing_pixels() -> None:
    experiment = MSImagingExperiment(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        feature_data=MassDataFrame([100.0, 200.0]),
        pixel_data=PositionDataFrame(
            {"x": [1, 2, 1], "y": [1, 1, 2], "run": ["r1"] * 3}
        ),
    )

    image = slice_image(experiment, i=0)
    kept_dimensions = slice_image(experiment, i=0, drop=False)
    query_result = slice_image(experiment, query="mz == 200")

    assert image.shape == (2, 2)
    np.testing.assert_allclose(image[0], [1.0, 2.0])
    assert np.isnan(image[1, 1])
    assert kept_dimensions.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(query_result[0], [4.0, 5.0])
