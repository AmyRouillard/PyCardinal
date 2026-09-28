"""Statistical and machine-learning models for spectral imaging data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.cross_decomposition import PLSRegression  # type: ignore[import-untyped]
from sklearn.decomposition import NMF as SklearnNMF  # type: ignore[import-untyped]
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]
from sklearn.preprocessing import (  # type: ignore[import-untyped]
    LabelEncoder,
    StandardScaler,
)

from pycardinal.core.imaging_data import MSImagingExperiment
from pycardinal.spatial import find_neighbors, spatial_weights


def _observations(value: Any) -> NDArray[np.float64]:
    if isinstance(value, MSImagingExperiment):
        matrix = value.intensity
        if sparse.issparse(matrix):
            matrix = cast(Any, matrix).toarray()
        return np.asarray(matrix, dtype=float).T
    if sparse.issparse(value):
        value = cast(Any, value).toarray()
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("model input must be a two-dimensional matrix")
    return matrix


def _feature_names(value: Any, count: int) -> NDArray[Any]:
    if isinstance(value, MSImagingExperiment):
        return np.asarray(value.mz, dtype=float)
    return np.arange(count)


def _prepare_x(
    value: Any, center: bool, scale: bool
) -> tuple[NDArray[np.float64], StandardScaler | None]:
    matrix = _observations(value)
    if not center and not scale:
        return matrix, None
    scaler = StandardScaler(with_mean=center, with_std=scale)
    return scaler.fit_transform(matrix), scaler


def _new_x(value: Any, scaler: StandardScaler | None) -> NDArray[np.float64]:
    matrix = _observations(value)
    return matrix if scaler is None else scaler.transform(matrix)


@dataclass
class SpatialPCA:
    """Principal-component scores, loadings, and fitted PCA state."""

    scores: NDArray[np.float64]
    loadings: NDArray[np.float64]
    sdev: NDArray[np.float64]
    model: Any
    scaler: StandardScaler | None
    feature_names: NDArray[Any]

    @property
    def x(self) -> NDArray[np.float64]:
        """Return observation scores."""
        return self.scores

    @property
    def rotation(self) -> NDArray[np.float64]:
        """Return feature loadings."""
        return self.loadings

    def predict(self, newdata: Any) -> NDArray[np.float64]:
        """Project new observations into the fitted PCA space."""
        return np.asarray(
            self.model.transform(_new_x(newdata, self.scaler)), dtype=float
        )


@dataclass
class SpatialNMF:
    """Nonnegative matrix-factorization scores and loadings."""

    scores: NDArray[np.float64]
    loadings: NDArray[np.float64]
    model: Any
    feature_names: NDArray[Any]

    @property
    def x(self) -> NDArray[np.float64]:
        """Return nonnegative observation scores."""
        return self.scores

    def predict(self, newdata: Any) -> NDArray[np.float64]:
        """Transform new observations using the fitted NMF model."""
        return np.asarray(
            self.model.transform(np.maximum(_observations(newdata), 0.0)), dtype=float
        )


@dataclass
class SpatialPLS:
    """Partial-least-squares regression or classification result."""

    scores: NDArray[np.float64]
    loadings: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    model: Any
    encoder: LabelEncoder | None
    feature_names: NDArray[Any]
    ncomp: int
    fitted_response: NDArray[np.float64]

    def predict(
        self,
        newdata: Any | None = None,
        ncomp: int | Sequence[int] | None = None,
        type: str = "response",
        simplify: bool = True,
    ) -> Any:
        """Predict responses or classes using one or more components."""
        matrix = self.model.x_scores_ if newdata is None else _observations(newdata)
        components = self.ncomp if ncomp is None else ncomp
        values = (
            [int(component) for component in components]
            if isinstance(components, (list, tuple, np.ndarray))
            else [int(cast(Any, components))]
        )
        predictions = [
            self._predict_one(matrix, int(cast(Any, component)), type)
            for component in values
        ]
        if len(predictions) == 1 or simplify is False:
            return predictions[0] if len(predictions) == 1 else predictions
        return np.stack(predictions, axis=-1)

    def _predict_one(self, matrix: Any, component: int, type: str) -> Any:
        if type not in {"response", "class"}:
            raise ValueError("type must be 'response' or 'class'")
        if matrix is self.model.x_scores_:
            response = self.fitted_response
        else:
            response = self.model.predict(matrix)
        if response.ndim == 2 and response.shape[1] > component:
            response = response[:, :component]
        if type == "class":
            if self.encoder is None:
                raise ValueError("class predictions require a categorical response")
            indices = np.argmax(response, axis=1)
            return self.encoder.inverse_transform(indices)
        return response

    def fitted(self, type: str = "response") -> Any:
        """Return fitted responses or classes for training data."""
        return self._predict_one(self.model.x_scores_, self.ncomp, type)

    def top_features(self, n: float = np.inf, sort_by: str = "vip") -> pd.DataFrame:
        """Rank features using fitted model loadings."""
        return top_features(self, n=n, sort_by=sort_by)


@dataclass
class SpatialOPLS(SpatialPLS):
    """Orthogonal partial-least-squares result."""

    def coef(self) -> NDArray[np.float64]:
        """Return feature-by-response regression coefficients."""
        return self.coefficients

    def residuals(self) -> NDArray[np.float64]:
        """Return training response residuals."""
        return np.asarray(
            self.model._y - self.model.predict(self.model._X), dtype=float
        )


@dataclass
class SpatialFastmap:
    """Spatially smoothed low-dimensional projection result."""

    scores: NDArray[np.float64]
    model: SpatialPCA
    feature_names: NDArray[Any]

    @property
    def x(self) -> NDArray[np.float64]:
        """Return projected observation scores."""
        return self.scores

    def predict(self, newdata: Any, **_: Any) -> NDArray[np.float64]:
        """Project new observations with the fitted model."""
        return self.model.predict(newdata)


@dataclass
class SpatialKMeans:
    """K-means cluster labels, centers, and feature correlations."""

    cluster: NDArray[np.intp]
    centers: NDArray[np.float64]
    correlation: NDArray[np.float64]
    feature_names: NDArray[Any]
    model: Any

    def predict(self, newdata: Any) -> NDArray[np.intp]:
        """Assign new observations to fitted clusters."""
        return np.asarray(self.model.predict(_observations(newdata)) + 1, dtype=np.intp)

    def top_features(
        self, n: float = np.inf, sort_by: str = "correlation"
    ) -> pd.DataFrame:
        """Rank features by cluster correlation."""
        return top_features(self, n=n, sort_by=sort_by)


@dataclass
class SpatialShrunkenCentroids:
    """Supervised or unsupervised shrunken-centroid result."""

    cluster: NDArray[np.intp]
    centers: NDArray[np.float64]
    feature_names: NDArray[Any]
    encoder: LabelEncoder | None
    model: Any

    def predict(self, newdata: Any) -> NDArray[Any]:
        """Predict class labels for new observations."""
        labels = self.model.predict(_observations(newdata))
        return (
            np.asarray(self.encoder.inverse_transform(labels))
            if self.encoder
            else np.asarray(labels + 1)
        )

    def fitted(self, type: str = "class") -> Any:
        """Return fitted classes or probabilities."""
        if type == "class":
            return self.predict_from_observations(self.model._X)
        return self.model.predict_proba(self.model._X)

    def predict_from_observations(self, matrix: NDArray[np.float64]) -> NDArray[Any]:
        """Predict labels from an observation-by-feature matrix."""
        labels = self.model.predict(matrix)
        return (
            np.asarray(self.encoder.inverse_transform(labels))
            if self.encoder
            else np.asarray(labels + 1)
        )

    def top_features(
        self, n: float = np.inf, sort_by: str = "statistic"
    ) -> pd.DataFrame:
        """Rank features by between-class center differences."""
        return top_features(self, n=n, sort_by=sort_by)


@dataclass
class SpatialDGMM:
    """Per-feature Gaussian-mixture segmentation result."""

    classes: NDArray[np.intp]
    probabilities: NDArray[np.float64]
    models: list[GaussianMixture]
    feature_names: NDArray[Any]

    def predict(self, newdata: Any) -> NDArray[np.intp]:
        """Predict one class label per selected feature and observation."""
        matrix = _observations(newdata)
        output = np.empty((matrix.shape[0], len(self.models)), dtype=np.intp)
        for index, model in enumerate(self.models):
            output[:, index] = model.predict(matrix[:, index : index + 1]) + 1
        return output

    def log_lik(self) -> float:
        """Return the summed fitted-data log likelihood."""
        return float(
            sum(model.score(model._X) * len(model._X) for model in self.models)
        )


@dataclass
class MeansTest:
    """Regression coefficients and fitted models from a means test."""

    statistics: pd.DataFrame
    models: list[Any]


@dataclass
class ContrastTest:
    """Estimated contrasts derived from a means test."""

    contrasts: pd.DataFrame


@dataclass
class SpatialCV:
    """Fold-level scores and optional fitted models from cross-validation."""

    scores: list[pd.DataFrame]
    average: pd.DataFrame
    models: list[Any] | None


def PCA(
    x: Any, ncomp: int = 3, center: bool = True, scale: bool = False, **kwargs: Any
) -> SpatialPCA:
    """Fit principal components to observations by features."""
    matrix, scaler = _prepare_x(x, center, scale)
    components = min(int(ncomp), matrix.shape[0], matrix.shape[1])
    model = SklearnPCA(n_components=components, **kwargs).fit(matrix)
    return SpatialPCA(
        scores=model.transform(matrix),
        loadings=model.components_.T,
        sdev=np.sqrt(model.explained_variance_),
        model=model,
        scaler=scaler,
        feature_names=_feature_names(x, matrix.shape[1]),
    )


def NMF(
    x: Any,
    ncomp: int = 3,
    method: str = "als",
    random_state: int | None = 0,
    **kwargs: Any,
) -> SpatialNMF:
    """Fit nonnegative matrix factorization with the selected solver."""
    if method not in {"als", "mult"}:
        raise ValueError("method must be 'als' or 'mult'")
    matrix = np.maximum(_observations(x), 0.0)
    solver = "mu" if method == "mult" else "cd"
    model = SklearnNMF(
        n_components=ncomp, solver=solver, random_state=random_state, **kwargs
    ).fit(matrix)
    return SpatialNMF(
        scores=model.transform(matrix),
        loadings=model.components_.T,
        model=model,
        feature_names=_feature_names(x, matrix.shape[1]),
    )


def _encode_response(y: ArrayLike) -> tuple[NDArray[np.float64], LabelEncoder | None]:
    values = np.asarray(y)
    if values.dtype.kind in {"U", "S", "O", "b"}:
        encoder = LabelEncoder().fit(values)
        labels = encoder.transform(values)
        return np.eye(len(encoder.classes_))[labels], encoder
    return np.asarray(values, dtype=float).reshape(len(values), -1), None


def PLS(
    x: Any,
    y: ArrayLike,
    ncomp: int = 3,
    method: str = "nipals",
    center: bool = True,
    scale: bool = False,
    **kwargs: Any,
) -> SpatialPLS:
    """Fit a supervised PLS baseline for regression or classification."""
    if method not in {"nipals", "simpls", "kernel1", "kernel2"}:
        raise ValueError("unsupported PLS method")
    matrix, scaler = _prepare_x(x, center, scale)
    response, encoder = _encode_response(y)
    components = min(int(ncomp), matrix.shape[1], matrix.shape[0] - 1)
    model = PLSRegression(n_components=components, scale=False, **kwargs).fit(
        matrix, response
    )
    model._X = matrix
    model._y = response
    model.x_scores_ = model.transform(matrix)
    fitted = model.predict(matrix)
    return SpatialPLS(
        scores=model.x_scores_,
        loadings=model.x_loadings_,
        coefficients=np.asarray(model.coef_).T,
        model=model,
        encoder=encoder,
        feature_names=_feature_names(x, matrix.shape[1]),
        ncomp=components,
        fitted_response=fitted,
    )


def OPLS(
    x: Any,
    y: ArrayLike,
    ncomp: int = 3,
    retx: bool = True,
    center: bool = True,
    scale: bool = False,
    **kwargs: Any,
) -> SpatialOPLS:
    """Fit the available OPLS-compatible supervised baseline."""
    result = PLS(x, y, ncomp=max(1, int(ncomp)), center=center, scale=scale, **kwargs)
    return SpatialOPLS(**result.__dict__)


def spatial_fastmap(
    x: Any,
    coord: Any = None,
    r: float = 1,
    ncomp: int = 3,
    weights: str = "gaussian",
    neighbors: Any = None,
    transpose: bool = True,
    niter: int = 10,
    **kwargs: Any,
) -> SpatialFastmap:
    """Fit a neighborhood-smoothed low-dimensional spatial projection."""
    if isinstance(x, MSImagingExperiment):
        data = x
        coordinates = x.coord if coord is None else coord
        neighbor_indices = (
            find_neighbors(coordinates, r=r) if neighbors is None else neighbors
        )
        weight_lists = cast(
            list[NDArray[np.float64]],
            spatial_weights(
                x,
                coord=coordinates,
                r=r,
                neighbors=cast(Any, neighbor_indices),
                weights=weights,
            ),
        )
        matrix = _observations(x)
        smoothing = np.zeros((len(weight_lists), len(weight_lists)))
        for index, local in enumerate(weight_lists):
            indices = np.asarray(cast(Any, neighbor_indices[index]), dtype=np.intp)
            local_values = np.asarray(local, dtype=float)
            smoothing[index, indices] = local / max(
                float(np.sum(local_values)), np.finfo(float).eps
            )
        matrix = smoothing @ matrix
    else:
        data = x
        matrix = _observations(x)
    model = PCA(matrix, ncomp=ncomp, center=True, scale=False)
    return SpatialFastmap(model.scores, model, _feature_names(data, matrix.shape[1]))


def spatial_kmeans(
    x: Any,
    coord: Any = None,
    r: float = 1,
    k: int | Sequence[int] = 2,
    ncomp: int | None = None,
    weights: str = "gaussian",
    neighbors: Any = None,
    transpose: bool = True,
    niter: int = 10,
    centers: bool = True,
    correlation: bool = True,
    random_state: int | None = 0,
    **kwargs: Any,
) -> SpatialKMeans | list[SpatialKMeans]:
    """Cluster observations with K-means for one or more cluster counts."""
    values = (
        [int(value) for value in k]
        if isinstance(k, (list, tuple, np.ndarray))
        else [int(cast(Any, k))]
    )
    matrix = _observations(x)
    results: list[SpatialKMeans] = []
    for number in values:
        model = KMeans(
            n_clusters=number, n_init=10, random_state=random_state, **kwargs
        ).fit(matrix)
        corr = np.empty((matrix.shape[1], number), dtype=float)
        for feature in range(matrix.shape[1]):
            for cluster in range(number):
                indicator = (model.labels_ == cluster).astype(float)
                value = np.corrcoef(matrix[:, feature], indicator)[0, 1]
                corr[feature, cluster] = 0.0 if not np.isfinite(value) else abs(value)
        results.append(
            SpatialKMeans(
                model.labels_ + 1,
                model.cluster_centers_,
                corr,
                _feature_names(x, matrix.shape[1]),
                model,
            )
        )
    return results[0] if len(results) == 1 else results


def spatial_shrunken_centroids(
    x: Any,
    y: ArrayLike | None = None,
    coord: Any = None,
    r: float = 1,
    k: int = 2,
    s: float = 0,
    weights: str = "gaussian",
    neighbors: Any = None,
    bags: Any = None,
    priors: Any = None,
    init: Any = None,
    threshold: float = 0.01,
    niter: int = 10,
    random_state: int | None = 0,
    **kwargs: Any,
) -> SpatialShrunkenCentroids:
    """Fit a supervised or unsupervised shrunken-centroid baseline."""
    matrix = _observations(x)
    encoder: LabelEncoder | None = None
    if y is None:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(matrix)
        labels = model.labels_
    else:
        encoder = LabelEncoder().fit(np.asarray(y))
        labels = encoder.transform(np.asarray(y))
        model = KMeans(
            n_clusters=len(encoder.classes_), n_init=10, random_state=random_state
        ).fit(matrix)
        model.labels_ = labels
        model.cluster_centers_ = np.vstack(
            [
                matrix[labels == value].mean(axis=0)
                for value in range(len(encoder.classes_))
            ]
        )
    model._X = matrix
    centers = model.cluster_centers_.copy()
    if s:
        centers[np.abs(centers) < float(s)] = 0
    return SpatialShrunkenCentroids(
        labels + 1, centers, _feature_names(x, matrix.shape[1]), encoder, model
    )


def spatial_dgmm(
    x: Any,
    i: ArrayLike | None = None,
    coord: Any = None,
    r: float = 1,
    k: int = 2,
    groups: Any = None,
    weights: str = "gaussian",
    neighbors: Any = None,
    annealing: bool = True,
    compress: bool = True,
    random_state: int | None = 0,
    **kwargs: Any,
) -> SpatialDGMM:
    """Fit independent Gaussian mixtures for selected feature columns."""
    matrix = _observations(x)
    indices = (
        np.arange(matrix.shape[1]) if i is None else np.asarray(i, dtype=int).ravel()
    )
    models: list[GaussianMixture] = []
    classes = np.empty((matrix.shape[0], len(indices)), dtype=np.intp)
    probabilities = np.empty((matrix.shape[0], len(indices), k), dtype=float)
    for output, feature in enumerate(indices):
        model = GaussianMixture(
            n_components=k, random_state=random_state, **kwargs
        ).fit(matrix[:, feature : feature + 1])
        model._X = matrix[:, feature : feature + 1]
        models.append(model)
        classes[:, output] = model.predict(model._X) + 1
        probabilities[:, output, :] = model.predict_proba(model._X)
    return SpatialDGMM(
        classes, probabilities, models, _feature_names(x, matrix.shape[1])[indices]
    )


def means_test(
    x: Any,
    fixed: str,
    random: str | None = None,
    samples: Any = None,
    response: str = "intensity",
    reduced: str = "~1",
    use_lmer: bool = False,
    **kwargs: Any,
) -> MeansTest:
    """Fit a formula-based means model to per-pixel experiment summaries."""
    import statsmodels.formula.api as smf  # type: ignore[import-untyped]

    if not isinstance(x, MSImagingExperiment):
        raise TypeError("means_test currently requires an imaging experiment")
    data = x.pixel_data.copy()
    values = _observations(x).mean(axis=1)
    data[response] = values
    formula = f"{response} ~ {fixed.lstrip('~ ')}"
    model = smf.ols(formula, data=data).fit()
    statistic = pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": model.params.to_numpy(),
            "pvalue": model.pvalues.to_numpy(),
        }
    )
    return MeansTest(statistic, [model])


def contrast_test(
    fit: MeansTest,
    specs: str,
    method: str = "pairwise",
    emm_adjust: str = "none",
) -> ContrastTest:
    """Return simple pairwise contrasts from a fitted means test."""
    if method != "pairwise":
        raise ValueError("only pairwise contrasts are currently supported")
    if emm_adjust not in {"none", "bonferroni"}:
        raise ValueError("emm_adjust must be 'none' or 'bonferroni'")
    statistics = fit.statistics.copy()
    selected = statistics[statistics["term"].astype(str).str.contains(specs)]
    if selected.empty:
        raise ValueError(f"no fitted terms match {specs!r}")
    contrasts = selected.assign(contrast=selected["term"].astype(str))[
        ["contrast", "estimate", "pvalue"]
    ]
    if emm_adjust == "bonferroni":
        contrasts["pvalue"] = np.minimum(contrasts["pvalue"] * len(contrasts), 1.0)
    return ContrastTest(contrasts.reset_index(drop=True))


def segmentation_test(
    x: Any,
    fixed: str,
    random: str | None = None,
    samples: Any = None,
    class_: int = 1,
    response: str = "intensity",
    reduced: str = "~1",
) -> MeansTest:
    """Run a means test using the first spatial segmentation class."""
    return means_test(
        x,
        fixed=fixed,
        random=random,
        samples=samples,
        response=response,
        reduced=reduced,
    )


def cross_validate(
    fit_fn: Any,
    x: Any,
    y: ArrayLike | None = None,
    folds: ArrayLike | None = None,
    predict_fn: Any = None,
    keep_models: bool = False,
    **fit_kwargs: Any,
) -> SpatialCV:
    """Fit and score a model independently on each validation fold."""
    matrix = _observations(x)
    labels = np.asarray(folds if folds is not None else np.arange(matrix.shape[0]) % 2)
    scores: list[pd.DataFrame] = []
    models: list[Any] = []
    for fold in pd.unique(labels):
        train = labels != fold
        test = labels == fold
        train_x = matrix[train]
        test_x = matrix[test]
        train_y = None if y is None else np.asarray(y)[train]
        model = (
            fit_fn(train_x, train_y, **fit_kwargs)
            if train_y is not None
            else fit_fn(train_x, **fit_kwargs)
        )
        predictor = predict_fn or getattr(model, "predict")
        prediction = predictor(test_x)
        truth = None if y is None else np.asarray(y)[test]
        accuracy = (
            float(np.mean(prediction == truth)) if truth is not None else float("nan")
        )
        scores.append(
            pd.DataFrame(
                {"fold": [fold], "accuracy": [accuracy], "n": [int(np.sum(test))]}
            )
        )
        models.append(model)
    average = pd.concat(scores, ignore_index=True)
    return SpatialCV(scores, average, models if keep_models else None)


def top_features(fit: Any, n: float = np.inf, sort_by: str = "vip") -> pd.DataFrame:
    """Rank features for a supported Phase 6 result object."""
    names = np.asarray(getattr(fit, "feature_names", []))
    if hasattr(fit, "loadings"):
        values = np.linalg.norm(np.asarray(fit.loadings), axis=1)
    elif hasattr(fit, "centers"):
        values = np.ptp(np.asarray(fit.centers), axis=0)
    elif hasattr(fit, "correlation"):
        values = np.max(np.asarray(fit.correlation), axis=1)
    else:
        raise TypeError(f"unsupported fit type: {type(fit).__name__}")
    frame = pd.DataFrame({"feature": names, sort_by: values})
    limit = len(frame) if np.isinf(n) else max(0, int(n))
    return (
        frame.sort_values(sort_by, ascending=False, kind="mergesort")
        .head(limit)
        .reset_index(drop=True)
    )


__all__ = [
    "MeansTest",
    "ContrastTest",
    "OPLS",
    "NMF",
    "PCA",
    "PLS",
    "SpatialCV",
    "SpatialDGMM",
    "SpatialFastmap",
    "SpatialKMeans",
    "SpatialOPLS",
    "SpatialPLS",
    "SpatialPCA",
    "SpatialShrunkenCentroids",
    "cross_validate",
    "contrast_test",
    "means_test",
    "segmentation_test",
    "spatial_dgmm",
    "spatial_fastmap",
    "spatial_kmeans",
    "spatial_shrunken_centroids",
    "top_features",
]
