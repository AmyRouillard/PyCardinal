import numpy as np

from pycardinal import (
    NMF,
    OPLS,
    PCA,
    PLS,
    MassDataFrame,
    MSImagingExperiment,
    PositionDataFrame,
    contrast_test,
    cross_validate,
    means_test,
    spatial_dgmm,
    spatial_fastmap,
    spatial_kmeans,
    spatial_shrunken_centroids,
    top_features,
)


def make_experiment() -> MSImagingExperiment:
    matrix = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0, 2.0, 3.0],
            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
        ]
    )
    return MSImagingExperiment(
        matrix,
        feature_data=MassDataFrame([100.0, 200.0, 300.0, 400.0]),
        pixel_data=PositionDataFrame(
            {
                "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "y": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "run": ["a", "a", "a", "b", "b", "b"],
            }
        ),
    )


def test_pca_nmf_and_top_features() -> None:
    experiment = make_experiment()

    pca = PCA(experiment, ncomp=2)
    nmf = NMF(experiment, ncomp=2, method="als")

    assert pca.scores.shape == (6, 2)
    assert pca.loadings.shape == (4, 2)
    assert pca.predict(experiment).shape == (6, 2)
    assert nmf.scores.shape == (6, 2)
    assert nmf.loadings.shape == (4, 2)
    assert nmf.predict(experiment).shape == (6, 2)
    assert len(top_features(pca, n=2)) == 2


def test_pls_opls_and_predictions() -> None:
    experiment = make_experiment()
    labels = np.array(["A", "A", "A", "B", "B", "B"])

    pls = PLS(experiment, labels, ncomp=2)
    opls = OPLS(experiment, labels, ncomp=1)

    assert pls.scores.shape == (6, 2)
    assert pls.predict(experiment, type="class").shape == (6,)
    assert pls.fitted(type="class").shape == (6,)
    assert opls.coef().shape[0] == 4
    assert opls.residuals().shape[0] == 6
    assert len(pls.top_features(n=2)) == 2


def test_spatial_models_and_means_test() -> None:
    experiment = make_experiment()

    fastmap = spatial_fastmap(experiment, ncomp=2)
    kmeans = spatial_kmeans(experiment, k=2)
    centroids = spatial_shrunken_centroids(experiment, k=2)
    dgmm = spatial_dgmm(experiment, i=[0, 1], k=2)
    means = means_test(experiment, fixed="run")
    contrasts = contrast_test(means, specs="run")

    assert fastmap.scores.shape == (6, 2)
    assert kmeans.cluster.shape == (6,)
    assert centroids.cluster.shape == (6,)
    assert dgmm.classes.shape == (6, 2)
    assert means.statistics.shape[0] >= 2
    assert contrasts.contrasts.shape[0] >= 1
    assert centroids.predict(experiment).shape == (6,)
    assert dgmm.predict(experiment).shape == (6, 2)


def test_cross_validate_returns_fold_scores() -> None:
    experiment = make_experiment()
    labels = np.array([0, 0, 0, 1, 1, 1])

    result = cross_validate(
        lambda train_x, train_y: PLS(train_x, train_y, ncomp=1),
        experiment,
        labels,
        folds=labels,
    )

    assert len(result.scores) == 2
    assert result.average.shape[0] == 2
