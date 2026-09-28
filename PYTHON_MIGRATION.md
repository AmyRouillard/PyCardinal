# Cardinal → Python Migration Record

This document records the phased direct port from the R/Bioconductor **Cardinal**
package to the pure-Python **`pycardinal`** implementation. It preserves the
original phase goals, decisions, validation evidence, and known limitations as a
historical migration record. The original Cardinal project remains the
behavioral and numerical reference.

Companion document: [PYTHON_API_REFERENCE.md](PYTHON_API_REFERENCE.md) lists
every function/class that needs a Python equivalent, with parameters.
Current release work is tracked separately in
[DEPLOYMENT_READINESS_PLAN.md](DEPLOYMENT_READINESS_PLAN.md).

## Migration principles

1. **Don't break the oracle.** Keep the R package intact until each Python
   module has been validated against it numerically (same inputs → same
   outputs within tolerance). Only remove R code in the final cutover phase.
2. **Test-driven port.** Every ported function gets a `pytest` test before or
  immediately after conversion, using small deterministic synthetic data.
  Private imzML files are optional for manual validation, never required by CI.
3. **One vertical slice at a time.** Prefer "read → process → analyze →
   write" working end-to-end on a tiny example over completing one layer
   fully before starting the next.
4. **Numerical parity over API parity.** The Python API should be idiomatic
   Python (snake_case, keyword args, numpy arrays), not a literal transliteration
   of R's S4 dispatch — but every R function's *behavior* needs a documented
   Python counterpart (see API reference).

## Historical target layout

```
pycardinal/
  pyproject.toml
  src/pycardinal/
    __init__.py
    io/
      imzml.py            # readImzML / writeImzML
      analyze.py           # readAnalyze / writeAnalyze
      common.py             # readMSIData / writeMSIData dispatch
    core/
      spectra_arrays.py    # SpectraArrays equivalent
      metadata.py          # PositionDataFrame / MassDataFrame (pandas-backed)
      imaging_data.py      # SpectralImagingData / SpectralImagingArrays
      imaging_experiment.py# SpectralImagingExperiment / MSImagingExperiment
      msimaging_arrays.py  # MSImagingArrays
      processing_queue.py  # addProcessing / process / reset (deferred ops)
    processing/
      normalize.py
      smooth.py
      baseline.py           # reduceBaseline
      recalibrate.py
      peaks.py              # peakPick / peakAlign / peakProcess
      binning.py            # bin / estimateDomain / estimateReferenceMz|Peaks
    spatial/
      neighbors.py          # findNeighbors
      weights.py            # spatialWeights
      distances.py          # spatialDists
    stats/
      pca.py
      nmf.py
      pls.py                # PLS / OPLS
      fastmap.py            # spatialFastmap
      kmeans.py             # spatialKMeans
      shrunken_centroids.py # spatialShrunkenCentroids
      dgmm.py               # spatialDGMM
      means_test.py         # meansTest / contrastTest / segmentationTest
      cross_validate.py     # crossValidate
    features.py             # features / pixels / subset / sliceImage / colocalized
    summarize.py             # summarizeFeatures / summarizePixels
    simulate.py              # simulateSpectra / simulateImage / presetImageDef
    plotting/
      image.py
      spectra.py
    roi.py                    # selectROI / makeFactor (interactive, matplotlib)
  tests/
    test_io.py
    test_processing.py
    test_stats.py
    test_simulate.py
    ...
  docs/
    api/                      # generated from docstrings (mkdocs/sphinx)
```

---

## Phase 0 — Project scaffolding

- [x] Create `pycardinal` package skeleton (`pyproject.toml`, `src/` layout, `pytest.ini`).
- [x] Declare core dependencies:
  - `numpy`, `scipy`, `pandas` — array/dataframe backbone.
  - `pyimzml` — imzML parsing (mirrors `CardinalIO::parseImzML`).
  - `scikit-learn` — PCA/NMF baseline algorithms (will be replaced/augmented with custom IRLBA-style solvers where needed).
  - `statsmodels` — mixed-effects models for `meansTest` (mirrors `nlme::lme`/`lmerTest`).
  - `matplotlib` — plotting.
  - `joblib` or `concurrent.futures` — parallelism (mirrors `BiocParallel`).
  - `numba` (optional) — hot-loop acceleration for peak picking/smoothing.
  - `pytest`, `pytest-benchmark`, `hypothesis` — testing.
- [x] Keep private datasets out of version control. The default suite uses simulated data and never reads local datasets.
- [x] Set up CI (lint via `ruff`, type-check via `mypy`, run `pytest`).
- [x] Set up the MkDocs documentation site, self-contained testing guidance, API skeleton, and documentation conventions.

**Exit criteria:** `pip install -e .` works; `pytest` runs (even with 0 tests).

---

## Phase 1 — Core data structures

Port the original Cardinal class hierarchy:

| R class | Python equivalent |
|---|---|
| `XDataFrame` / `XDFrame` | `pandas.DataFrame` subclass with a `.keys` dict of "key columns" (e.g. coordinate or mz columns) |
| `PositionDataFrame` | `PositionDataFrame(XDFrame)` — requires `x`,`y`(,`z`) + `run` columns |
| `MassDataFrame` | `MassDataFrame(XDFrame)` — requires `mz` column, enforces sorted order |
| `SpectraArrays` | `SpectraArrays` — dict-like container of equal-shaped arrays (dense `np.ndarray` or `scipy.sparse`, or lazy/file-backed) |
| `SpectralImagingData` (virtual) | `SpectralImagingData` — abstract base: owns `spectra_data`, `pixel_data`, `processing` queue |
| `SpectralImagingArrays` / `MSImagingArrays` | ragged/"processed"-mode: list-of-arrays per pixel (variable-length spectra) |
| `SpectralImagingExperiment` / `MSImagingExperiment` | "continuous"-mode: shared `mz` axis, dense/sparse matrix `features x pixels` |

- [x] Implement `SpectraArrays` (get/set arrays by name, enforce equal shape).
- [x] Implement `PositionDataFrame`/`MassDataFrame` on top of `pandas.DataFrame`.
- [x] Implement `SpectralImagingExperiment`/`MSImagingExperiment` with:
  - `.mz`, `.intensity`, `.coord`, `.run`, `.pixel_data`, `.feature_data`
  - `__getitem__` supporting feature/pixel subsetting (`obj[rows, cols]`)
  - `.centroided`, `.experiment_data` (metadata dict mirroring `ImzMeta`)
- [x] Implement `MSImagingArrays` (list of variable-length `(mz, intensity)` pairs per pixel).
- [x] Implement the **deferred processing queue**: `add_processing(fn, label, **meta)`, `process(...)`, `reset(...)` — mirrors the original Cardinal processing behavior. Each queued step is `(label, callable, kwargs)`; `process()` applies them all in order to every spectrum (chunked/parallel).
- [x] Unit tests: construct experiments from numpy arrays, round-trip subsetting, verify processing-queue apply order.
- [x] Documentation: describe class invariants, feature/pixel axis conventions, metadata, and processing-queue behavior with small examples.

**Exit criteria:** Can build an `MSImagingExperiment` from raw numpy arrays and inspect it; processing queue applies user functions correctly.

---

## Phase 2 — I/O (imzML / Analyze)

Mirrors the original Cardinal read/write behavior.

- [x] `read_imzml(path, memory=True, mass_range=None, resolution=None, units="ppm", guess_max=1000, as_="auto", parse_only=False)` using `pyimzml.ImzMLParser`. Detect representation from IMS CV terms and return `MSImagingExperiment` or `MSImagingArrays`.
- [x] Implement `convert_arrays_to_experiment` / `convert_experiment_to_arrays` (nearest-bin conversion to/from a shared `mz` axis).
- [x] Add `read_analyze(path, ...)` placeholder for Analyze 7.5 (`.img`/`.hdr`/`.t2m`); it explicitly raises `NotImplementedError`.
- [x] `read_msi_data(path, **kwargs)` — dispatch imzML and ibd paths; identify Analyze as unsupported.
- [x] `write_imzml(obj, path, bundle=True, ...)` — writes continuous or processed imzML from either class, with one-run/1-based-coordinate validation.
- [x] `write_msi_data(obj, path, **kwargs)` — dispatches imzML outputs and identifies unsupported Analyze outputs.
- [x] **I/O verification**: synthetic imzML pairs test representation metadata, pixel/spectrum count, conversion, and processed/continuous write/read round trips. A private-file manual check also confirmed processed-spectrum parsing.
- [x] Documentation: specify supported formats, continuous/processed detection, conversion options, bundle paths, and read/write examples, including current eager-only and checksum limitations.

**Exit criteria:** Generated imzML fixtures load and round-trip with matching `mz`/intensity arrays (within floating point tolerance).

---

## Phase 3 — Spectral processing pipeline

Mirrors the original Cardinal spectral-processing behavior.

- [x] `normalize(obj, method="tic"|"rms"|"reference", scale=None, ref=None)` — queues rescaling.
- [x] `smooth(obj, method="gaussian"|"bilateral"|"adaptive"|"diff"|"guide"|"pag"|"sgolay"|"ma", **kwargs)`.
- [x] `reduce_baseline(obj, method="locmin"|"hull"|"snip"|"median", **kwargs)`.
- [x] `recalibrate(obj, ref, method="locmax"|"dtw"|"cow", tolerance=None, units="ppm")`.
- [x] `peak_pick(obj, ref=None, method="diff"|"sd"|"mad"|"quantile"|"filter"|"cwt", snr=2, type_="height"|"area", tolerance=None, units="ppm")`.
- [x] `peak_align(obj, ref=None, binratio=2, tolerance=None, units="ppm")`.
- [x] `bin_spectra(obj, ref=None, method="sum"|"mean"|"max"|"min", resolution=None, tolerance=None, mass_range=None)`; interpolation method names raise `NotImplementedError`.
- [x] `peak_process(obj, ref=None, method=..., snr=2, sample_size=None, filter_freq=True, ...)` — combined pick+align convenience wrapper.
- [x] `estimate_domain(xlist, width="median", units="relative")`, `estimate_reference_mz(obj, ...)`, `estimate_reference_peaks(obj, snr=2, method="diff")`.
- [x] Deferred functions queue through the Phase-1 processing queue; alignment, binning, and peak-process materialize results eagerly.
- [x] Added synthetic tests for preprocessing, reference extraction, peak alignment/binning, and the end-to-end pipeline.
- [x] Documentation: document parameters, units, eager/deferred behavior, approximation limits, and reproducible examples.

**Exit criteria:** A full `normalize → smooth → reduce_baseline → peak_pick → peak_align` pipeline runs on simulated data and produces a peak-picked `MSImagingExperiment`.

---

## Phase 4 — Feature/pixel utilities

Mirrors the original Cardinal feature, image, and spatial behavior.

- [x] `features(obj, **conditions)` / `pixels(obj, **conditions)` — return matching row/col indices from expressions over `feature_data`/`pixel_data` (use `DataFrame.query` or boolean kwargs).
- [x] `subset_features(obj, **conditions)`, `subset_pixels(obj, **conditions)`, `subset(obj, select=None, subset=None)`.
- [x] `slice_image(obj, i=None, run=None, simplify=True, drop=True, **conditions)` — returns a numpy array/list of 2D image slices.
- [x] `find_neighbors(coord_or_obj, r=1, groups=None, metric="maximum", p=2, matrix=False)` — `scipy.spatial.cKDTree`-based neighbor search per group/run.
- [x] `spatial_weights(x, coord=None, r=1, neighbors=None, weights="gaussian"|"adaptive", sd=None, matrix=False)`.
- [x] `spatial_dists(...)`.
- [x] `colocalized(obj, i=None, mz=None, ref=None, threshold="median", n=np.inf, sort_by="cor")`.
- [x] Port `tests/testthat/test-findNeighbors.R`, `test-spatial.R`, `test-colocalized.R`, `test-sliceImage.R`.
- [x] Documentation: explain feature/pixel selection, image shape and coordinate conventions, neighbor metrics, and colocalization measures.

**Exit criteria:** Can slice ion images by m/z and compute colocalization / spatial neighbor lists matching R output on synthetic data.

---

## Phase 5 — Summaries

Mirrors the original Cardinal summary behavior.

- [x] `summarize_features(obj, stat="mean", groups=None)`, `summarize_pixels(obj, stat={"tic": "sum"}, groups=None)`.
- [x] `row_stats`/`col_stats` helpers (`min,max,prod,sum,mean,var,sd,any,all,nnzero`) over dense/sparse data.
- [x] Port the summary behavior from `tests/testthat/test-summarize.R` with deterministic dense/sparse tests.
- [x] Documentation: supported summary statistics, grouping semantics, output columns, and NaN handling are documented in the Python API reference.

---

## Phase 6 — Statistics & machine learning

Mirrors `R/stats-*.R`. These are the highest-value/highest-risk conversions — validate numerically against R outputs on the same simulated dataset (fixed seed) before trusting them on real data.

- [x] `PCA(x, ncomp=3, center=True, scale=False)` — sklearn PCA baseline with scores, loadings, sdev, and prediction.
- [x] `NMF(x, ncomp=3, method="als"|"mult")` — sklearn coordinate-descent and multiplicative-update baselines.
- [x] `PLS(...)` and `OPLS(...)` — supervised regression/classification baseline with prediction, fitted values, coefficients, residuals, and feature ranking.
- [x] `spatial_fastmap(...)` — neighborhood-smoothed PCA projection baseline.
- [x] `spatial_kmeans(...)` — KMeans clustering with feature/cluster correlations and list-of-k support.
- [x] `spatial_shrunken_centroids(...)` — supervised or unsupervised centroid baseline with shrinkage and prediction.
- [x] `spatial_dgmm(...)` — per-feature Gaussian-mixture segmentation baseline.
- [x] `means_test(...)`, `contrast_test(...)`, and `segmentation_test(...)` — statsmodels-backed means and contrast baseline.
- [x] `cross_validate(...)` — fold-based fit/predict harness with optional model retention.
- [x] `top_features(...)` — generic feature-ranking dispatch for implemented result types.
- [ ] Port the R statistics/cross-validation tests for numerical parity; the current tests cover deterministic shapes and behavioral invariants.
- [x] Documentation: describe the implemented baseline model surface and its numerical-parity limitations.

**Exit criteria:** `PCA`, `spatial_kmeans`, and `spatial_shrunken_centroids` on the same simulated image produce clusters/scores matching R to reasonable numerical tolerance (exact bitwise match is not expected/required, especially for RNG-driven methods — document the required tolerance).

---

## Phase 7 — Simulation utilities (needed for tests)

Mirrors the original Cardinal simulation behavior. These utilities provide the generated spectra/images used by tests and examples, avoiding a required external dataset.

- [x] `simulate_spectra(n=1, npeaks=50, mz=None, intensity=None, ...)`.
- [x] `preset_image_def(preset=1, nrun=1, npeaks=30, dim=(20,20), ...)` and `simulate_image(pixel_data=None, feature_data=None, preset=None, ...)`.
- [x] `add_shape(pixel_data, center, size, shape="circle"|"square", name=None)`.
- [x] Use `numpy.random.Generator` with explicit seeding to keep tests reproducible.
- [x] Documentation: describe simulation parameters, seeded reproducibility, preset designs, and intended use for tests/examples.

---

## Phase 8 — Visualization

Mirrors the original Cardinal plotting behavior and `vizi_*` concepts.

- [x] `plot_spectra(obj, i=None, superpose=False, xlim=None, ...)` via `matplotlib`.
- [x] `plot_image(obj, feature=None, i=None, superpose=False, scale=False, ...)` for shared-domain ion images.
- [x] Per-model `image_model(...)`/`plot_model(...)` helpers for PCA scores/scree, loadings/centers, and cluster maps.
- [x] `select_roi(obj, mode="region"|"pixels")`, `make_factor(**named_masks, ordered=False)` with deterministic polygon/point selection and categorical masks.
- [x] Documentation: plotting and ROI behavior, headless testing, and interactive limitations are documented in the API reference.

**Exit criteria:** Can reproduce the example plots from the `.Rd` `\examples{}` blocks (visual smoke test, no pixel-perfect requirement).

---

## Phase 9 — Synthetic and optional integration testing

- [x] Replace automatic real-file fixtures with deterministic simulation tests; the default suite has no dependency on external imzML/ibd files.
- [x] `tests/test_io.py` creates temporary imzML/ibd pairs from simulated experiments and tests metadata parsing and continuous/processed round trips.
- [x] `tests/test_processing.py`: run normalization, smoothing, baseline reduction, recalibration, peak picking/alignment, binning, and an end-to-end preprocessing pipeline on simulated data.
- [x] `tests/test_stats.py`: run the Phase 6 model baselines on deterministic simulated-style images and assert result shapes and behavioral invariants.
- [x] Keep the default simulation and temporary-imzML tests small and fast; no slow marker is currently needed.
- [x] Optional cross-language comparison harness: `scripts/cross_language_compare.py` writes a fixed-seed Python JSON artifact and compares it with an externally generated R JSON artifact when R/Cardinal is available.
- [x] Documentation: state that no private data is required for pytest and describe how to use private files manually without adding them to the test suite.

**Exit criteria:** The default `pytest` run is self-contained and passes in CI without any imzML/ibd files.

---

## Phase 10 — Documentation

- [x] Add NumPy-style docstrings to public functions/classes and generate API pages with `mkdocs` + `mkdocstrings`.
- [x] Keep [PYTHON_API_REFERENCE.md](PYTHON_API_REFERENCE.md) as the canonical parameter reference alongside the generated API pages.
- [x] Rewrite `README.md` for the Python package with installation and `read_imzml` → `process` → `PCA` → `plot_image` quickstarts.
- [x] Review and publish the user guide, API reference, migration notes, and installation instructions in the MkDocs site.

---

## Phase 11 — Packaging

- [x] Keep the project installable via `pip install -e .` / `pip install git+...`.
- [x] Finalize package metadata, build artifacts, supported Python versions, and release checks.
- [x] Add packaging/installation CI validation for source and built distributions.

---

## Phase 12 — Cutover

Only after Phases 1–11 are validated against generated datasets, documentation,
packaging, and the R test suite's behavior is matched (or intentional
deviations documented):

- [x] Mark the original R package as historical and point the root `README.md` to the Python package.
- [x] Preserve the original R implementation and documentation outside the maintained Python surface.
- [x] Remove the root `NAMESPACE`/R-package build surface; CI now validates only the Python package and its distributions.
- [x] Publish the R-to-Python migration/cutover notes, compatibility notes, and the supported Python installation path.

---

## Migration Status

The migration phases are complete at baseline level. The package has deterministic
coverage for core data, I/O, processing, summaries, statistics, plotting, and
simulation. Exact R numerical-parity tests remain open for execution in an
R/Cardinal environment. The original Cardinal project remains the numerical
oracle. Release hardening and deployment decisions are tracked in
[DEPLOYMENT_READINESS_PLAN.md](DEPLOYMENT_READINESS_PLAN.md).
