import numpy as np
from scipy import sparse

from pycardinal import find_neighbors, spatial_dists, spatial_weights


def grid_coordinates() -> np.ndarray:
    return np.array([(x, y) for y in range(3) for x in range(3)], dtype=float)


def test_find_neighbors_uses_inclusive_radius_and_metric() -> None:
    coordinates = grid_coordinates()

    maximum = find_neighbors(coordinates, r=1)
    euclidean = find_neighbors(coordinates, r=1, metric="euclidean")

    np.testing.assert_array_equal(maximum[0], [0, 1, 3, 4])
    np.testing.assert_array_equal(euclidean[0], [0, 1, 3])


def test_find_neighbors_respects_groups_and_sparse_matrix_mode() -> None:
    coordinates = grid_coordinates()
    groups = np.array(["left"] * 3 + ["right"] * 6)

    neighbors = find_neighbors(coordinates, r=1, groups=groups)
    matrix = find_neighbors(coordinates, r=1, groups=groups, matrix=True)

    np.testing.assert_array_equal(neighbors[0], [0, 1])
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (9, 9)
    assert matrix[0, 0] == matrix[0, 1] == 1
    assert matrix[0, 3] == 0


def test_spatial_weights_match_gaussian_kernel_and_adaptive_mode() -> None:
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0]])
    neighbors = find_neighbors(coordinates, r=1)
    gaussian = spatial_weights(coordinates, r=1, neighbors=neighbors)
    adaptive = spatial_weights(
        np.array([[0.0], [10.0]]),
        coord=coordinates,
        r=1,
        neighbors=neighbors,
        weights="adaptive",
    )
    matrix = spatial_weights(coordinates, r=1, neighbors=neighbors, matrix=True)

    expected = np.exp(-(1.0**2) / (2 * 0.75**2))
    np.testing.assert_allclose(gaussian[0], [1.0, expected])
    assert adaptive[0][1] < gaussian[0][1]
    assert sparse.isspmatrix_csr(matrix)
    np.testing.assert_allclose(matrix.toarray()[0, :2], gaussian[0])


def test_spatial_dists_returns_neighborhood_weighted_observation_distances() -> None:
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    values = np.array([[0.0], [2.0], [4.0]])
    target = np.array([[1.0]])
    neighbors = find_neighbors(coordinates, r=1, metric="euclidean")
    neighborhood_weights = [np.ones(len(indices)) for indices in neighbors]

    distances = spatial_dists(
        values,
        target,
        coord=coordinates,
        neighbors=neighbors,
        neighbors_weights=neighborhood_weights,
    )

    np.testing.assert_allclose(distances[:, 0], [1.0, 5.0 / 3.0, 2.0])


def test_spatial_dists_accepts_feature_by_pixel_experiment_orientation() -> None:
    from pycardinal import MassDataFrame, MSImagingExperiment, PositionDataFrame

    experiment = MSImagingExperiment(
        np.array([[0.0, 2.0, 4.0]]),
        feature_data=MassDataFrame([100.0]),
        pixel_data=PositionDataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]}),
    )

    result = spatial_dists(
        experiment,
        np.array([[1.0]]),
        r=1,
        neighbors=find_neighbors(experiment, r=1, metric="euclidean"),
        neighbors_weights=[np.ones(2), np.ones(3), np.ones(2)],
    )

    np.testing.assert_allclose(result[:, 0], [1.0, 5.0 / 3.0, 2.0])
