# Python API Reference (target design for `pycardinal`)

This is a parameter-level reference for the full Python API directly ported
from the R/Bioconductor **Cardinal** package. It describes available APIs and
planned equivalents; check [docs/api-reference.md](docs/api-reference.md)
for the current API surface. Python signatures use snake_case and
Python-native defaults; the "R source" column points to the original behavior
and documentation.

Conventions used below:
- `obj` / `x` — a spectral imaging dataset (`MSImagingExperiment`, `MSImagingArrays`, or base `SpectralImagingExperiment`/`Arrays`).
- Params marked **(deferred)** are queued and only applied when `process()` is called.
- `verbose`, `chunkopts`, `BPPARAM` in R (progress/chunking/parallel backend) collapse to a single `n_jobs=None, chunk_size=None, verbose=False` triplet in Python, using `joblib`/`concurrent.futures` instead of `BiocParallel`.

---

## 1. Core classes (`pycardinal.core`)

### `SpectraArrays`
Dict-like container of equal-length array-likes (dense, sparse, or lazy).
- `SpectraArrays(data: dict[str, ArrayLike] | None = None)`
- `.names -> list[str]`
- `__getitem__(name) / __setitem__(name, value)`

### `PositionDataFrame(pandas.DataFrame)`
Pixel-level metadata. Required columns: `x`, `y` (optional `z`), `run`.
- `PositionDataFrame(coord: DataFrame | dict, run: array-like | None = None, **cols)`
- `.coord -> DataFrame` (the `x`/`y`/`z` columns)
- `.run -> pandas.Categorical`

### `MassDataFrame(pandas.DataFrame)`
Feature-level metadata. Required column: `mz` (sorted ascending).
- `MassDataFrame(mz: array-like, **cols)`
- `.mz -> np.ndarray`

### `SpectralImagingData` (abstract base)
- `.spectra_data -> SpectraArrays`
- `.pixel_data -> PositionDataFrame`
- `.processing -> list[ProcessingStep]` (the deferred-processing queue)

### `SpectralImagingExperiment(SpectralImagingData)`
Shared-domain ("continuous") dataset: features × pixels matrix.
- `.feature_data -> DataFrame`
- `.spectra(name="intensity") -> np.ndarray | scipy.sparse matrix`
- `__getitem__((rows, cols))`, `.shape`, `len(obj)` (n pixels)

### `MSImagingExperiment(SpectralImagingExperiment)`
```python
MSImagingExperiment(
    spectra_data: SpectraArrays | dict | ArrayLike = None,
    feature_data: MassDataFrame = None,
    pixel_data: PositionDataFrame = None,
    experiment_data: dict | None = None,
    centroided: bool | None = None,
    metadata: dict | None = None,
)
```
- `.mz -> np.ndarray` (get/set)
- `.intensity -> np.ndarray` (get/set)
- `.centroided -> bool | None` (get/set); `.is_centroided() -> bool`
- `.experiment_data -> dict | None` (mirrors `ImzMeta`)
- R source: [man/MSImagingExperiment-class.Rd](legacy-r/man/MSImagingExperiment-class.Rd), [legacy-r/R/AllClasses.R](legacy-r/R/AllClasses.R), [legacy-r/R/methods-MSImagingExperiment.R](legacy-r/R/methods-MSImagingExperiment.R)

### `MSImagingArrays(SpectralImagingArrays)`
Ragged ("processed") dataset: one variable-length `(mz, intensity)` pair per pixel.
- `.mz -> list[np.ndarray]`, `.intensity -> list[np.ndarray]` (get/set per-pixel)
- `.centroided`, `.continuous`, `.experiment_data`
- R source: [man/MSImagingArrays-class.Rd](legacy-r/man/MSImagingArrays-class.Rd)

### Processing queue (`pycardinal.core.processing_queue`)
| Python | R source | Notes |
|---|---|---|
| `add_processing(obj, fn, label, metadata=None, **fn_kwargs) -> obj` | `addProcessing()` — [man/process.Rd](legacy-r/man/process.Rd) | `fn(intensity, mz, **fn_kwargs) -> intensity or (mz, intensity)` |
| `process(obj, spectra="intensity", index="mz", domain=None, outfile=None, n_jobs=None, chunk_size=None, verbose=False) -> obj` | `process()` | Applies all queued steps in order; writes imzML if `outfile` given |
| `reset(obj) -> obj` | `reset()` | Clears the queue |

---

## 2. I/O (`pycardinal.io`)

| Python | R source | Key params |
|---|---|---|
| `read_msi_data(file, **kwargs)` | `readMSIData()` | dispatches by extension (`.imzml`/`.ibd` → imzML, `.img`/`.hdr`/`.t2m` → Analyze) |
| `read_imzml(file, memory=True, check=False, mass_range=None, resolution=None, units="ppm", guess_max=1000, as_="auto", parse_only=False, verbose=False)` | `readImzML()` — [man/readMSIData.Rd](legacy-r/man/readMSIData.Rd) | Available eager reads; `memory=False` and checksum `check=True` raise `NotImplementedError`; `as_ in {"auto","experiment","arrays"}` |
| `read_analyze(file, memory=False, as_="auto", verbose=False)` | `readAnalyze()` | Legacy Analyze 7.5 format |
| `convert_arrays_to_experiment(obj, mz=None, mass_range=None, resolution=None, units="ppm", guess_max=1000, tolerance=None)` | `convertMSImagingArrays2Experiment()` | Available nearest-bin conversion to sparse shared `mz`; varying axes need `mz` or `resolution` |
| `convert_experiment_to_arrays(obj)` | `convertMSImagingExperiment2Arrays()` | Available; returns nonzero per-pixel values |
| `write_msi_data(obj, file, **kwargs)` | `writeMSIData()` | Available dispatch for imzML paths; Analyze recognized but not available |
| `write_imzml(obj, file, bundle=True, verbose=False)` | `writeImzML()` — [man/writeMSIData.Rd](legacy-r/man/writeMSIData.Rd) | Available continuous/processed output; one run, positive integer coordinates |
| `write_analyze(obj, file, ...)` | `writeAnalyze()` | Legacy format |

**Parameter notes**
- `mass_range`: `(low, high)` tuple restricting the shared m/z axis.
- `resolution`/`units`: bin width for building the shared `mz` axis (`units in {"ppm","mz"}`).
- `guess_max`: number of spectra sampled to infer mass range/resolution when not given.

---

## 3. Spectral processing (`pycardinal.processing`)

All of these queue work onto `obj.processing` unless noted **(eager)**; call
`process(obj)` to actually apply them.

| Python | R source | Params |
|---|---|---|
| `normalize(obj, method="tic", scale=None, ref=None, tolerance=None, units="ppm")` **(deferred)** | [man/normalize.Rd](legacy-r/man/normalize.Rd) | Available `tic`, `rms`, `reference`; callback scales to target; zero denominator leaves signal unchanged |
| `smooth(obj, method="gaussian", width=5, sigma=None, sigma_range=None, polyorder=2, iterations=5, kappa=None, step=0.2, epsilon=1e-3)` **(deferred)** | [man/smooth.Rd](legacy-r/man/smooth.Rd) | Available `gaussian`, `bilateral`, `adaptive`, `diff`, `guide`, `pag`, `sgolay`, `ma`; widths are sample counts |
| `reduce_baseline(obj, method="locmin", window=31, iterations=40)` **(deferred)** | [man/reduceBaseline.Rd](legacy-r/man/reduceBaseline.Rd) | Available `locmin`, `hull`, `snip`, `median`; output clipped at zero |
| `recalibrate(obj, ref, method="locmax", tolerance=None, units="ppm")` **(deferred)** | [man/recalibrate.Rd](legacy-r/man/recalibrate.Rd) | Available `locmax`, `dtw`, `cow` names using local-maximum anchors and monotone interpolation; approximate algorithms |
| `peak_pick(obj, ref=None, method="diff", snr=2, type_="height", tolerance=None, units="ppm", **options)` **(deferred)** | [man/peakPick.Rd](legacy-r/man/peakPick.Rd) | Available `diff`, `sd`, `mad`, `quantile`, `filter`, `cwt`; `type_` is `height` or `area` |
| `peak_align(obj, ref=None, method="diff", snr=2, tolerance=None, units="ppm", binratio=2, n_jobs=None)` **(eager)** | [man/peakAlign.Rd](legacy-r/man/peakAlign.Rd) | applies queued steps first, detects/merges peaks, and returns a sparse shared-axis experiment |
| `bin_spectra(obj, ref=None, method="sum", resolution=None, tolerance=None, units="ppm", mass_range=None)` **(eager)** | [man/bin.Rd](legacy-r/man/bin.Rd) | Available aggregation methods: `sum`, `mean`, `max`, `min`; interpolation methods currently raise `NotImplementedError` |
| `peak_process(obj, ref=None, method="diff", snr=2, type_="height", tolerance=None, units="ppm", sample_size=None, binratio=2, filter_freq=True, n_jobs=None, **peak_options)` **(eager, orchestrates pick+align)** | [man/peakProcess.Rd](legacy-r/man/peakProcess.Rd) | reference sampling, extraction, alignment, and frequency filtering |
| `estimate_domain(xlist, width="median", units="relative")` | [man/estimateDomain.Rd](legacy-r/man/estimateDomain.Rd) | Available `median`, `min`, `max`, `mean` gap summaries |
| `estimate_reference_mz(obj, width="median", units="ppm")` | same file | returns the existing axis for shared-domain data; estimates one from ragged m/z vectors |
| `estimate_reference_peaks(obj, method="diff", snr=2)` | same file | peak-picks the mean spectrum |

**`peak_process` behavior:** If `sample_size` is supplied without `ref`, it samples evenly spaced spectra (fraction if `<1`, otherwise count), estimates a merged reference, extracts peaks across the full object, aligns, then applies `filter_freq`. Without `sample_size`, it peak-picks the full input and aligns the detections. This is a practical implementation, not an exact numeric match to R/matter.

---

## 4. Feature / pixel utilities (`pycardinal.features`)

| Python | R source | Params |
|---|---|---|
| `features(obj, **conditions) -> np.ndarray[int]` | [man/features.Rd](legacy-r/man/features.Rd) | e.g. `features(obj, mz=800.5)` |
| `pixels(obj, **conditions) -> np.ndarray[int]` | [man/pixels.Rd](legacy-r/man/pixels.Rd) | e.g. `pixels(obj, x=3, y=3)` |
| `subset_features(obj, **conditions)` | [man/subset.Rd](legacy-r/man/subset.Rd) | boolean expression over `feature_data` |
| `subset_pixels(obj, **conditions)` | same file | boolean expression over `pixel_data` |
| `subset(obj, select=None, subset=None)` | same file | `select`/`subset` are boolean masks or callables |
| `slice_image(obj, i=None, run=None, simplify=True, drop=True, **conditions)` | [man/sliceImage.Rd](legacy-r/man/sliceImage.Rd) | `i`: feature indices; `**conditions` forwarded to `features()` |
| `colocalized(obj, i=None, mz=None, ref=None, threshold="median", n=np.inf, sort_by="cor")` | [man/colocalized.Rd](legacy-r/man/colocalized.Rd) | `sort_by in {"cor","MOC","M1","M2","Dice","none"}` |
| `coregister(...)` | same file | image co-registration helper |

---

## 5. Spatial utilities (`pycardinal.spatial`)

| Python | R source | Params |
|---|---|---|
| `find_neighbors(x, r=1, groups=None, metric="maximum", p=2, matrix=False)` | [man/findNeighbors.Rd](legacy-r/man/findNeighbors.Rd) | `metric in {"euclidean","maximum","manhattan","minkowski"}` |
| `spatial_weights(x, coord=None, r=1, neighbors=None, weights="gaussian", sd=None, matrix=False, byrow=True)` | [man/spatialWeights.Rd](legacy-r/man/spatialWeights.Rd) | `weights in {"gaussian","adaptive"}`; `sd` default `((2r+1)/4)` |
| `spatial_dists(...)` | [man/spatialDists.Rd](legacy-r/man/spatialDists.Rd) | pairwise spatial distance helper |

---

## 6. Summaries (`pycardinal.summarize`)

| Python | R source | Params |
|---|---|---|
| `summarize_features(obj, stat="mean", groups=None)` | [man/summarize.Rd](legacy-r/man/summarize.Rd) | Adds overall or grouped summaries across pixels; grouped columns use `group.stat` names and support `na_rm`. |
| `summarize_pixels(obj, stat={"tic": "sum"}, groups=None)` | same file | Adds overall or grouped summaries across features; `stat` maps output column name → stat and supports `na_rm`. |
| `row_stats(x, stat, ...)` / `col_stats(x, stat, ...)` | same file | Dense and SciPy sparse reductions for `min`, `max`, `prod`, `sum`, `mean`, `var`, `sd`, `any`, `all`, and `nnzero`. |

---

## 7. Statistics & machine learning (`pycardinal.stats`)

The Phase 6 model surface is available as a practical sklearn/statsmodels
baseline. Result objects expose fitted values, prediction, and feature-ranking
helpers where applicable. Exact numerical parity with Cardinal's `matter`
algorithms, full multiple-instance-learning behavior, and production-scale
chunked fitting remain validation work.

### `PCA(x, ncomp=3, center=True, scale=False) -> SpatialPCA`
R source: [man/SpatialPCA.Rd](legacy-r/man/SpatialPCA.Rd)
- `.scores`, `.loadings`, `.sdev`
- `.predict(newdata) -> scores`
- `.plot(type="rotation"|"scree"|"x")`, `.image(type="x")`

### `NMF(x, ncomp=3, method="als") -> SpatialNMF`
R source: [man/SpatialNMF.Rd](legacy-r/man/SpatialNMF.Rd)
- `method in {"als","mult"}`
- `.predict(newdata)`, `.plot(type="activation"|"x")`, `.image(type="x")`

### `PLS(x, y, ncomp=3, method="nipals", center=True, scale=False, bags=None) -> SpatialPLS`
### `OPLS(x, y, ncomp=3, retx=True, center=True, scale=False, bags=None) -> SpatialOPLS`
R source: [man/SpatialPLS.Rd](legacy-r/man/SpatialPLS.Rd)
- `method in {"nipals","simpls","kernel1","kernel2"}`
- `.fitted(type="response"|"class")`
- `.predict(newdata, ncomp=None, type="response"|"class", simplify=True)`
- `.top_features(n=np.inf, sort_by="vip"|"coefficients")`
- `OPLS` additionally exposes `.coef()`, `.residuals()`

### `spatial_fastmap(x, coord=None, r=1, ncomp=3, weights="gaussian", neighbors=None, transpose=True, niter=10) -> SpatialFastmap`
R source: [man/SpatialFastmap.Rd](legacy-r/man/SpatialFastmap.Rd)
- `.predict(newdata, weights=None, r=None, neighbors=None)`

### `spatial_kmeans(x, coord=None, r=1, k=2, ncomp=None, weights="gaussian", neighbors=None, transpose=True, niter=10, centers=True, correlation=True) -> SpatialKMeans`
R source: [man/SpatialKMeans.Rd](legacy-r/man/SpatialKMeans.Rd)
- `ncomp` defaults to `max(k)`; first projects via `spatial_fastmap`, then k-means.
- `.top_features(n=np.inf, sort_by="correlation")`
- `.plot(type="correlation"|"centers")`, `.image(type="cluster")`

### `spatial_shrunken_centroids(x, y=None, coord=None, r=1, k=2, s=0, weights="gaussian", neighbors=None, bags=None, priors=None, center=None, transpose=None, init=None, threshold=0.01, niter=10) -> SpatialShrunkenCentroids`
R source: [man/SpatialShrunkenCentroids.Rd](legacy-r/man/SpatialShrunkenCentroids.Rd)
- If `y` given → supervised (classification); else → clustering with `k` classes.
- `s`: sparsity/shrinkage parameter (can be a list/array to fit a path of models).
- `bags`: multiple-instance-learning groups.
- `.fitted(type="response"|"class")`, `.predict(newdata, weights=None, r=None, neighbors=None)`
- `.log_lik()`, `.top_features(n=np.inf, sort_by="statistic"|"centers")`
- `.plot(type="statistic"|"centers")`, `.image(type="probability"|"class")`

### `spatial_dgmm(x, i=None, coord=None, r=1, k=2, groups=None, weights="gaussian", neighbors=None, annealing=True, compress=True, byrow=False) -> SpatialDGMM`
R source: [man/SpatialDGMM.Rd](legacy-r/man/SpatialDGMM.Rd)
- Fits a spatially-smoothed Gaussian mixture **per feature** (`i`: which features/rows to segment).
- `groups`: independent segmentation per group/run (needed for `means_test`).
- `.log_lik()`, `.plot(i=1, type="density", layout=None, free="")`, `.image(i=1, type="class", layout=None, free="")`

### `means_test(x, fixed, random=None, samples=None, response="intensity", reduced="~1", use_lmer=False, na_rm=True, class_=1) -> MeansTest`
R source: [man/MeansTest.Rd](legacy-r/man/MeansTest.Rd)
- `x`: `SpectralImagingExperiment` or `SpatialDGMM` result.
- `fixed`/`random`: R-style formula strings (e.g. `"~ condition"`), parsed via `patsy`/`statsmodels.formula.api`.
- `use_lmer=False` → likelihood-ratio test against `reduced` model (mirrors `nlme::lme`); `use_lmer=True` → REML fit only, no test (use `contrast_test` for post-hoc).
- `.top_features(n=np.inf, sort_by="statistic")`, `.plot(i=1, type="boxplot", show_obs=True, fill=False, layout=None)`

### `contrast_test(fit, specs, method="pairwise", emm_adjust="none") -> ContrastTest`
R source: same file
- `specs`: factor name(s) or formula for estimated marginal means.
- `method in {"pairwise","trt.vs.ctrl","poly", <custom dict>}`
- `emm_adjust in {"none","bonferroni","tukey","fdr",...}`

### `segmentation_test(x, fixed, random=None, samples=None, class_=1, response="intensity", reduced="~1") -> MeansTest`
R source: same file — convenience wrapper: `spatial_dgmm` + `means_test`.

### `cross_validate(fit_fn, x, y=None, folds=None, predict_fn=None, keep_models=False, train_process=peak_process, train_kwargs=None, test_process=peak_process, test_kwargs=None) -> SpatialCV`
R source: [man/SpatialCV.Rd](legacy-r/man/SpatialCV.Rd)
- `folds`: fold label per pixel (defaults to `run(x)`).
- `.fitted(type="response"|"class")`, `.image(i=1, type="response"|"class", layout=None, free="")`

### `top_features(fit, n=np.inf, sort_by=...)`
Generic dispatcher across all result types above.

---

## 8. Simulation (`pycardinal.simulate`)

| Python | R source | Key params |
|---|---|---|
| `simulate_spectra(n=1, npeaks=50, mz=None, intensity=None, from_=None, to=None, by=400, sdpeaks=None, sdpeakmult=0.2, sdnoise=0.1, sdmz=10, resolution=1000, fmax=0.5, baseline=0, decay=10, units="ppm", centroided=False, random_state=None)` | [man/simulateSpectra.Rd](legacy-r/man/simulateSpectra.Rd) | Available; returns `MassDataFrame` with an `intensity` column or one intensity column per spectrum |
| `simulate_image(pixel_data=None, feature_data=None, preset=None, from_=None, to=None, by=400, sdrun=1, sdpixel=1, spcorr=0.3, sar=False, resolution=1000, fmax=0.5, units="ppm", centroided=False, continuous=True, random_state=None, **preset_kwargs)` | same file | Available; returns `MSImagingExperiment` or `MSImagingArrays` |
| `add_shape(pixel_data, center, size, shape="circle", name=None)` | same file | Available; `shape in {"circle","square"}`; returns copied metadata |
| `preset_image_def(preset=1, nrun=1, npeaks=30, dim=(20,20), peakheight=e, peakdiff=e, sdsample=0.2, jitter=True, random_state=None)` | same file | Available nine deterministic-design presets; seed via `random_state` |

---

## 9. Plotting & ROI (`pycardinal.plotting`, `pycardinal.roi`)

The Phase 8 baseline uses Matplotlib and supports headless test rendering. The
plotting helpers return Matplotlib axes so callers can continue customizing
figures. Interactive ROI picking is intentionally represented by deterministic
polygon/point inputs in the Python API; GUI-specific selectors can be layered on
later without changing the returned boolean-mask contract.

| Python | R source | Params |
|---|---|---|
| `plot_spectra(obj, i=None, superpose=False, xlim=None, ylim=None, **kwargs)` | [man/plot-spectra.Rd](legacy-r/man/plot-spectra.Rd) | Available for shared-domain and ragged MS imaging objects; returns axes. |
| `plot_image(obj, feature=None, i=None, superpose=False, scale=False, **kwargs)` | [man/plot-image.Rd](legacy-r/man/plot-image.Rd) | Available for shared-domain ion images; returns axes. |
| `plot_model(fit, type="scores", **kwargs)` / `image_model(fit, type="x", **kwargs)` | Phase 6 result methods | Available baseline helpers for model scores, scree, loadings, centers, and classes. |
| `select_roi(obj, mode="region", polygon=None, points=None, tolerance=0.5)` | [man/selectROI.Rd](legacy-r/man/selectROI.Rd) | Deterministic polygon/point masks; `mode in {"region","pixels"}`. |
| `make_factor(ordered=False, **named_masks) -> pandas.Categorical` | same file | Available first-match combination of named boolean masks. |

---

## 10. Options / configuration (`pycardinal.config`)

Mirrors [legacy-r/R/options.R](legacy-r/R/options.R) `getCardinal*`/`setCardinal*` functions —
collapse into a single settings object instead of global getter/setter pairs:

```python
from pycardinal.config import settings

settings.n_jobs = 4          # was getCardinalBPPARAM/setCardinalBPPARAM
settings.verbose = True       # was getCardinalVerbose/setCardinalVerbose
settings.chunk_size = 1000    # was getCardinalChunksize/setCardinalChunksize
settings.n_chunks = None      # was getCardinalNChunks/setCardinalNChunks
```

---

## Appendix: R API Coverage

### Available

- Core dataset containers, metadata frames, matrix slicing, and deferred queue:
    `MSImagingExperiment`, `MSImagingArrays`, `SpectralImagingExperiment`,
    `SpectralImagingArrays`, `XDataFrame`, `PositionDataFrame`, `MassDataFrame`,
    `SpectraArrays`, `process`, `addProcessing`, and `reset` equivalents.
- imzML reading, writing, format dispatch, and experiment/array conversion.
- Spectral preprocessing: normalization, smoothing, baseline reduction,
    recalibration, peak picking/alignment/process, supported bin aggregation,
    and reference-axis/peak estimation.
- Simulation: spectra, images, shapes, and the nine preset designs.
- Visualization and ROI: Matplotlib spectra/image/model plots, deterministic
    polygon/point ROI masks, and named-mask categorical factors.

### Remaining or Partial

- Analyze 7.5 reader/writer are exported placeholders that raise
    `NotImplementedError`.
- `bin_spectra` interpolation modes (`linear`, `cubic`, `gaussian`, `lanczos`)
    are recognized but not available; sum/mean/max/min are available.
- Feature/pixel query and ROI slicing, colocalization/coregistration, spatial
    neighbors/weights/distances, summaries, baseline statistical/ML models,
    plotting, and deterministic ROI helpers are available; exact numerical
    parity and GUI-specific selectors remain in
    [PYTHON_MIGRATION_PLAN.md](PYTHON_MIGRATION_PLAN.md).
- `vizi_*`, palettes/facets/layers, and Cardinal global options are not
    available yet.

Use `NAMESPACE` as the source checklist when expanding coverage. The detailed
sections above distinguish current implementation from the target API.
