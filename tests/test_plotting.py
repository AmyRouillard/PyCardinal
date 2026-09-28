import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pycardinal import (
    MassDataFrame,
    MSImagingExperiment,
    PositionDataFrame,
    image_model,
    make_factor,
    plot_image,
    plot_model,
    plot_spectra,
    select_roi,
)
from pycardinal.stats import PCA, spatial_kmeans


def make_experiment() -> MSImagingExperiment:
    return MSImagingExperiment(
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
            ]
        ),
        feature_data=MassDataFrame([100.0, 200.0]),
        pixel_data=PositionDataFrame(
            {
                "x": [0.0, 1.0, 0.0, 1.0],
                "y": [0.0, 0.0, 1.0, 1.0],
            }
        ),
    )


def test_plot_spectra_and_image_return_axes() -> None:
    experiment = make_experiment()
    figure, axes = plt.subplots()

    spectra_axes = plot_spectra(experiment, i=[0, 1], ax=axes)
    image_axes = plot_image(experiment, i=0, ax=axes)

    assert spectra_axes is axes
    assert image_axes is axes
    assert len(axes.lines) == 2
    assert len(axes.images) == 1
    plt.close(figure)


def test_model_plot_helpers_return_axes() -> None:
    experiment = make_experiment()
    pca = PCA(experiment, ncomp=2)
    kmeans = spatial_kmeans(experiment, k=2)
    figure, axes = plt.subplots()

    assert plot_model(pca, type="scores", ax=axes) is axes
    assert plot_model(pca, type="scree", ax=axes) is axes
    assert image_model(kmeans, type="cluster", ax=axes) is axes
    plt.close(figure)


def test_roi_masks_and_make_factor_are_deterministic() -> None:
    experiment = make_experiment()

    region = select_roi(
        experiment,
        mode="region",
        polygon=[[-0.1, -0.1], [1.1, -0.1], [1.1, 0.1], [-0.1, 0.1]],
    )
    pixels = select_roi(experiment, mode="pixels", points=[[0.0, 1.0]])
    factor = make_factor(A=region, B=pixels)

    np.testing.assert_array_equal(region, [True, True, False, False])
    np.testing.assert_array_equal(pixels, [False, False, True, False])
    assert list(factor.astype(object))[:3] == ["A", "A", "B"]
    assert pd.isna(factor.astype(object)[3])
    assert list(factor.categories) == ["A", "B"]
