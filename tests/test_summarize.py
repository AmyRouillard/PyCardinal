import numpy as np
import pytest
from scipy import sparse

from pycardinal import (
    MassDataFrame,
    MSImagingArrays,
    MSImagingExperiment,
    PositionDataFrame,
    col_stats,
    row_stats,
    summarize_features,
    summarize_pixels,
)


def make_experiment(matrix: object) -> MSImagingExperiment:
    return MSImagingExperiment(
        matrix,
        feature_data=MassDataFrame([100.0, 200.0, 300.0]),
        pixel_data=PositionDataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]}),
    )


def test_row_and_column_stats_support_nan_and_all_stat_names() -> None:
    matrix = np.array([[1.0, 2.0, np.nan], [0.0, 2.0, 4.0]])

    np.testing.assert_allclose(row_stats(matrix, "sum", na_rm=True), [3.0, 6.0])
    np.testing.assert_allclose(row_stats(matrix, "mean", na_rm=True), [1.5, 2.0])
    np.testing.assert_allclose(col_stats(matrix, "max", na_rm=True), [1.0, 2.0, 4.0])
    np.testing.assert_allclose(
        col_stats(matrix, "var", na_rm=True), [0.5, 0.0, np.nan], equal_nan=True
    )
    np.testing.assert_array_equal(row_stats(matrix, "nnzero", na_rm=True), [2, 2])
    np.testing.assert_array_equal(row_stats(matrix, "any", na_rm=True), [True, True])
    np.testing.assert_array_equal(row_stats(matrix, "all", na_rm=True), [True, False])


def test_sparse_stats_match_dense_stats() -> None:
    matrix = np.array([[1.0, 0.0, 3.0], [4.0, 5.0, 0.0]])
    sparse_matrix = sparse.csr_matrix(matrix)

    np.testing.assert_allclose(
        row_stats(sparse_matrix, "sum"), row_stats(matrix, "sum")
    )
    np.testing.assert_allclose(
        col_stats(sparse_matrix, "mean"), col_stats(matrix, "mean")
    )
    np.testing.assert_array_equal(col_stats(sparse_matrix, "nnzero"), [2, 1, 1])


def test_summarize_features_adds_grouped_columns() -> None:
    experiment = make_experiment(
        np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
    )

    result = summarize_features(experiment)
    result = summarize_features(result, groups=["A", "A", "B"])

    np.testing.assert_allclose(result.feature_data["mean"], [2.0, 4.0, 6.0])
    np.testing.assert_allclose(result.feature_data["A.mean"], [1.5, 3.0, 4.5])
    np.testing.assert_allclose(result.feature_data["B.mean"], [3.0, 6.0, 9.0])
    assert "mean" not in experiment.feature_data


def test_summarize_pixels_supports_named_stats_and_feature_groups() -> None:
    experiment = make_experiment(
        np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
    )

    result = summarize_pixels(experiment)
    result = summarize_pixels(
        result,
        stat={"tic": "sum", "average": "mean"},
        groups=["light", "light", "heavy"],
    )

    np.testing.assert_allclose(result.pixel_data["tic"], [6.0, 12.0, 18.0])
    np.testing.assert_allclose(result.pixel_data["light.tic"], [3.0, 6.0, 9.0])
    np.testing.assert_allclose(result.pixel_data["heavy.average"], [3.0, 6.0, 9.0])


def test_summary_rejects_ragged_data_and_invalid_groups() -> None:
    ragged = MSImagingArrays(
        mz=[np.array([100.0]), np.array([200.0, 201.0])],
        intensity=[np.array([1.0]), np.array([2.0, 3.0])],
        pixel_data=PositionDataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]}),
    )
    experiment = make_experiment(np.ones((3, 3)))

    with pytest.raises(TypeError, match="shared-domain"):
        summarize_features(ragged)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one label"):
        summarize_pixels(experiment, groups=["A"])
    with pytest.raises(ValueError, match="unsupported statistic"):
        row_stats(np.ones((2, 2)), "median")
