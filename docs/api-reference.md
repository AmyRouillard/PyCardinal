# API Reference

This page documents the current Python API for the direct Python port of the
R/Bioconductor Cardinal package. The complete migration
target, including planned functions and parameter mappings to R, is indexed in
`PYTHON_API_REFERENCE.md` at the repository root.

## Core Data

All current classes and queue functions are available from both `pycardinal`
and `pycardinal.core`.

### `SpectraArrays`

`SpectraArrays(data=None)` stores named NumPy, SciPy sparse, or sequence-backed
arrays. Values must have identical shapes; ragged per-pixel spectra use
sequences whose outer length is the pixel count.

- `data`: optional mapping from non-empty string names to array-like values.
- `.names`: stored names in insertion order.
- `arrays[name]`: retrieve or set a named array; assigning a differently shaped
	value raises `ValueError`.

### Metadata frames

- `PositionDataFrame(coord=None, run=None, **columns)` accepts a DataFrame, a
	mapping containing numeric `x` and `y` (optional `z`), or an `N x 2`/`N x 3`
	coordinate array. `run` defaults to `"run1"`. `.coord` returns coordinate
	columns and `.run` returns categorical run labels.
- `MassDataFrame(mz=None, **columns)` accepts an m/z vector, mapping, or
	DataFrame. It requires finite, nondecreasing m/z values. `.mz` returns the
	values as a NumPy array.
- `XDFrame`/`XDataFrame` are DataFrame bases with a `key_columns` mapping. The
	name `key_columns` avoids shadowing pandas' `.keys()` method.

### Imaging data

`MSImagingExperiment(spectra_data, *, feature_data, pixel_data,
experiment_data=None, centroided=None, processing=None, metadata=None)` holds
shared-domain spectra. The intensity matrix is always **features by pixels**:
`shape == (n_features, n_pixels)`. `feature_data` requires a sorted `mz`
column; `pixel_data` has one row per matrix column.

- `.mz` and `.intensity`: get or set the shared m/z axis and intensity matrix;
	replacements must keep the existing number of features and matrix shape.
- `.feature_data`, `.pixel_data`, `.coord`, `.run`, `.spectra_data`, and
	`.experiment_data`: access feature, pixel, and experiment metadata/data.
- `dataset[feature_selector, pixel_selector]`: return a subset preserving
	feature-by-pixel orientation and dense or sparse storage.
- `.is_centroided()`: returns `True` only when `centroided is True`.

`MSImagingArrays(mz=..., intensity=..., pixel_data=..., ...)` stores one
variable-length m/z array and intensity array per pixel. The two lists must
have equal outer lengths, and each corresponding pair must have equal-length
one-dimensional arrays. It also exposes `.centroided`, `.continuous`, and
`.experiment_data`.

### Deferred processing

```python
import numpy as np
from pycardinal import (
		MSImagingExperiment,
		MassDataFrame,
		PositionDataFrame,
		add_processing,
		process,
)

dataset = MSImagingExperiment(
		np.array([[1.0, 2.0], [3.0, 4.0]]),
		feature_data=MassDataFrame([100.0, 101.0]),
		pixel_data=PositionDataFrame({"x": [0, 1], "y": [0, 0]}),
)
queued = add_processing(
		dataset,
		lambda intensity, mz, scale: intensity * scale,
		"scale",
		scale=2.0,
)
processed = process(queued)
```

- `add_processing(obj, fn, label, metadata=None, **fn_kwargs)` returns a copy
	with the step appended. Each callback receives one pixel's
	`(intensity, mz, **fn_kwargs)` and returns either new intensity or
	`(new_mz, new_intensity)`.
- `process(obj, n_jobs=None, chunk_size=None, verbose=False)` runs queued
	steps in order and returns a new object with an empty queue. `chunk_size`
	must be positive; `n_jobs=0` is invalid. Shared-domain experiments require
	every pixel to return the same m/z axis. Callbacks may mutate their inputs;
	processing uses copies so the source dataset remains unchanged.
- `reset(obj)` returns a copy with queued steps removed and data unchanged.

## imzML I/O

### `read_imzml(file, *, memory=True, check=False, mass_range=None, resolution=None, units="ppm", guess_max=1000, as_="auto", parse_only=False, verbose=False)`

Reads the matching `.imzML` and `.ibd` pair using pyimzml. Representation is
detected from `IMS:1000030` (continuous) or `IMS:1000031` (processed).

- `memory=True`: spectra are currently loaded eagerly. `memory=False` raises
	`NotImplementedError` until file-backed access is available.
- `check=True`: currently raises `NotImplementedError`; checksum validation is
	not available yet.
- `as_="auto"`: continuous data returns `MSImagingExperiment`; processed data
	returns `MSImagingArrays`.
- `as_="arrays"` and `as_="experiment"` request a representation explicitly.
	Converting variable-length spectra to an experiment requires `mz=` or a
	`resolution`; a range can be supplied with `mass_range=(low, high)`.
- `parse_only=True` returns representation, coordinates, and metadata without
	reading the spectral arrays.

### Conversion and writing

- `convert_arrays_to_experiment(obj, *, mz=None, mass_range=None,
	resolution=None, units="ppm", guess_max=1000, tolerance=None)` maps ragged
	peaks to nearest shared-axis bins and returns a sparse intensity matrix.
	`units` is `"ppm"` or `"mz"`; `tolerance` uses the selected units.
- `convert_experiment_to_arrays(obj)` returns nonzero intensity values as
	per-pixel spectra.
- `write_imzml(obj, file, *, bundle=True, verbose=False)` writes imzML and its
	`.ibd` sidecar. A bundle path creates a directory named after `file`; use
	`bundle=False` to write at the requested `.imzML` path. Written coordinates
	must be positive integer (1-based) positions, and multiple runs are not yet
	supported.
- `read_msi_data(file, **kwargs)` and `write_msi_data(obj, file, **kwargs)`
	dispatch `.imzML`/`.ibd` paths to imzML. Analyze 7.5 extensions are
	recognized but currently raise `NotImplementedError`.

## Spectral Processing

Deferred functions return a copy with a processing step queued. Use
`pycardinal.process(dataset)` to apply queued operations in order; the source
dataset is left unchanged.

- `normalize(obj, method="tic", scale=None, ref=None, tolerance=None,
	units="ppm")` supports total-ion-current (`tic`), root-mean-square (`rms`),
	and reference-peak (`reference`) normalization.
- `smooth(obj, method="gaussian", width=5, sigma=None, sigma_range=None,
	polyorder=2, iterations=5, kappa=None, step=0.2, epsilon=1e-3)` supports
	`gaussian`, `bilateral`, `adaptive`, `diff`, `guide`, `pag`, `sgolay`, and
	`ma`. Width is measured in spectrum samples.
- `reduce_baseline(obj, method="locmin", window=31, iterations=40)` supports
	`locmin`, `hull`, `snip`, and `median`; corrected values are clipped at zero.
- `recalibrate(obj, ref, method="locmax", tolerance=None, units="ppm")`
	supports `locmax`, `dtw`, and `cow` names using detected local maxima and a
	monotone interpolation warp. These are practical approximations, not exact
	matter/R algorithms.
- `peak_pick(obj, ref=None, method="diff", snr=2, type_="height",
	tolerance=None, units="ppm", **options)` supports noise estimators
	`diff`, `sd`, `mad`, `quantile`, `filter`, and `cwt`; `type_` may be
	`height` or `area`.
- `peak_align(obj, ref=None, method="diff", snr=2, tolerance=None,
	units="ppm", binratio=2, n_jobs=None)` applies queued steps, aligns peaks,
	and returns a sparse `MSImagingExperiment` with feature metadata `count` and
	`freq`.
- `peak_process(obj, ref=None, method="diff", snr=2, type_="height",
	tolerance=None, units="ppm", sample_size=None, binratio=2,
	filter_freq=True, n_jobs=None, **options)` combines picking, alignment, and
	optional frequency filtering. A `sample_size` count or proportion estimates
	the reference from evenly spaced pixels.
- `bin_spectra(obj, ref=None, method="sum", resolution=None,
	tolerance=None, units="ppm", mass_range=None)` eagerly bins spectra using
	`sum`, `mean`, `max`, or `min`. Interpolation modes `linear`, `cubic`,
	`gaussian`, and `lanczos` currently raise `NotImplementedError`.
- `estimate_domain(xlist, width="median", units="relative")`,
	`estimate_reference_mz(obj, width="median", units="ppm")`, and
	`estimate_reference_peaks(obj, method="diff", snr=2)` provide axis and peak
	references for alignment. `estimate_reference_mz` returns an experiment's
	existing shared axis or estimates an axis from ragged spectra.

```python
from pycardinal import (
		normalize,
		peak_process,
		process,
		reduce_baseline,
		simulate_image,
		smooth,
)

image = simulate_image(preset=1, npeaks=12, dim=(8, 8), random_state=7)
image = normalize(image, method="tic")
image = smooth(image, method="gaussian", width=5)
image = reduce_baseline(image, method="locmin", window=15)
peaks = peak_process(image, method="sd", snr=2, filter_freq=False)
```

## Feature and Pixel Selection

`features(obj, query=None, **conditions)` and
`pixels(obj, query=None, **conditions)` return zero-based NumPy index arrays.
Queries are pandas expressions against the corresponding metadata frame; simple
conditions combine with logical AND. Exact keyword names perform equality (or
membership for a list), and suffixes `__eq`, `__ne`, `__lt`, `__le`, `__gt`,
`__ge`, `__in`, and `__notin` select a comparison. Callable conditions receive
the column Series and must return a boolean mask.

```python
from pycardinal import features, pixels, subset_features, subset_pixels

feature_indices = features(image, query="mz > 800", mz__le=1800)
pixel_indices = pixels(image, query="x >= 2", region="circle")
feature_subset = subset_features(image, mz__ge=800, mz__le=1800)
pixel_subset = subset_pixels(image, region="circle")
```

`subset(obj, select=None, subset=None)` applies positional feature and pixel
selectors to a shared-domain experiment. `subset_features` and `subset_pixels`
filter by metadata predicates. Pixel filtering also supports ragged
`MSImagingArrays`; ragged feature selection is undefined because each spectrum
can have its own m/z axis. Sparse intensity matrices and feature/pixel metadata
are preserved.

`slice_image(obj, i=None, run=None, simplify=True, drop=True, **conditions)`
returns selected shared-domain features as 2D ion-image arrays. Rows follow
ascending `y`, columns follow ascending `x`, and missing grid coordinates are
filled with NaN. `drop=False` retains feature and run dimensions. Three-dimensional
coordinates and duplicate coordinates within one run are currently unsupported.

`find_neighbors(coord_or_obj, r=1, groups=None, metric="maximum", p=2,
matrix=False)` returns zero-based neighbor index arrays, including each point
itself; `matrix=True` returns a SciPy CSR adjacency matrix. `metric="maximum"`
uses Chebyshev distance. `spatial_weights(x, coord=None, r=1, neighbors=None,
weights="gaussian", sd=None, matrix=False)` computes Gaussian spatial weights;
`weights="adaptive"` also weights by data-vector similarity. For an imaging
experiment, pixel coordinates and feature vectors are inferred.

`spatial_dists(x, y, coord=None, r=1, neighbors=None,
neighbors_weights=None, weights=None, byrow=True, metric="euclidean", p=2)`
returns center-by-target weighted distances. Imaging experiments are interpreted
as feature-by-pixel matrices automatically. The current implementation computes
dense pairwise distance results, so very large comparisons may need chunking in
a later increment.

`colocalized(obj, i=None, mz=None, ref=None, threshold="median", n=np.inf,
sort_by="cor")` ranks features against a reference feature, m/z value, or pixel
vector using correlation, MOC, M1, M2, or Dice scores.

## Summary Statistics

`row_stats(x, stat, na_rm=False)` and `col_stats(x, stat, na_rm=False)` reduce
rows or columns of dense NumPy arrays and SciPy sparse matrices. Supported
statistics are `min`, `max`, `prod`, `sum`, `mean`, `var`, `sd`, `any`, `all`,
and `nnzero`. Variance and standard deviation use sample degrees of freedom
(`ddof=1`); empty or single-value variance reductions return NaN. With
`na_rm=True`, NaN values are excluded before reduction.

`summarize_features(obj, stat="mean", groups=None, na_rm=False)` adds summaries
across pixels to `feature_data`. `summarize_pixels(obj, stat={"tic": "sum"},
groups=None, na_rm=False)` adds summaries across features to `pixel_data`.
`stat` may be a statistic name or a mapping from output column names to statistic
names. Group labels apply to the reduced axis and grouped columns use
`group.stat` names. The functions return a copied experiment, so the input
metadata is unchanged.

```python
from pycardinal import row_stats, summarize_features, summarize_pixels

feature_summary = summarize_features(image, groups=image.pixel_data["run"])
pixel_summary = summarize_pixels(image, stat={"tic": "sum", "mean": "mean"})
row_totals = row_stats(image.intensity, "sum")
```

## Statistics and Machine Learning

The Phase 6 baseline is available from `pycardinal.stats` and the package root:
`PCA`, `NMF`, `PLS`, `OPLS`, `spatial_fastmap`, `spatial_kmeans`,
`spatial_shrunken_centroids`, `spatial_dgmm`, `means_test`, `contrast_test`,
`segmentation_test`, `cross_validate`, and `top_features`. Models use the
existing NumPy/SciPy/scikit-learn/statsmodels stack and operate on imaging data
with pixels as observations and features as columns. Exact parity with Cardinal
`matter` algorithms, full multiple-instance-learning semantics, and chunked
large-data fitting are not yet certified.

## Simulation

All simulation functions use NumPy's `Generator`; pass `random_state` as an
integer to reproduce a generated design and spectra. Tests use these functions
and temporary imzML pairs, so pytest does not need private or downloaded data.

- `simulate_spectra(n=1, npeaks=50, mz=None, intensity=None, from_=None,
	to=None, by=400, sdpeaks=None, sdpeakmult=0.2, sdnoise=0.1, sdmz=10,
	resolution=1000, fmax=0.5, baseline=0, decay=10, units="ppm",
	centroided=False, random_state=None)` returns `MassDataFrame`: `mz` is the
	shared axis; one spectrum is in `intensity`, multiple spectra use
	`intensity_1`, `intensity_2`, and so on. Profile mode adds Gaussian peaks,
	baseline decay, and multiplicative noise; centroided mode returns peak
	intensities on the theoretical axis.
- `preset_image_def(preset=1, nrun=1, npeaks=30, dim=(20, 20),
	peakheight=np.e, peakdiff=np.e, sdsample=0.2, jitter=True,
	random_state=None)` returns a dictionary containing `pixel_data` and
	`feature_data`. Presets 1-8 generate 2D designs; preset 9 supports 3D.
- `add_shape(pixel_data, center, size, shape="circle", name=None)` returns
	copied pixel metadata with a boolean ROI column. A circle uses Euclidean
	radius; a square uses a half-width per coordinate dimension.
- `simulate_image(pixel_data=None, feature_data=None, preset=None, from_=None,
	to=None, by=400, sdrun=1, sdpixel=1, spcorr=0.3, sar=False,
	resolution=1000, fmax=0.5, units="ppm", centroided=False,
	continuous=True, random_state=None, **preset_kwargs)` returns
	`MSImagingExperiment` for continuous output or `MSImagingArrays` for
	processed output. A custom design needs numeric feature intensity columns
	with matching boolean region columns in `pixel_data`.

The generators are behaviorally useful for testing and examples but do not
reproduce Cardinal R's random stream bit-for-bit. `sar=True` uses sparse
nearest-neighbor smoothing; for numerical stability its effective spatial
coefficient is capped at 0.95.

## Plotting and ROI

`plot_spectra` and `plot_image` use Matplotlib and return axes for further
customization. `plot_model` and `image_model` provide baseline visualization of
Phase 6 result objects. `select_roi` accepts deterministic polygons or points
and returns a boolean pixel mask; `make_factor` combines named masks with
first-match precedence. These APIs are tested with the noninteractive `Agg`
backend. GUI-specific polygon selectors remain optional because they depend on
the active Matplotlib backend.

## Module Status

| Module | Status / contents |
|---|---|
| `pycardinal.core` | Available: data containers, metadata, and processing queue |
| `pycardinal.io` | Available: imzML read, convert, and write; Analyze remains a placeholder |
| `pycardinal.simulate` | Available: spectra, image, ROI-shape, and preset generators |
| `pycardinal.processing` | Available: normalization, smoothing, baseline reduction, recalibration, peak detection/alignment, binning, and reference estimation |
| `pycardinal.features` | Available: feature/pixel metadata selection, dataset subsetting, 2D ion-image slicing, and colocalization |
| `pycardinal.spatial` | Available: grouped neighbor search, sparse adjacency, distances, and Gaussian/adaptive spatial weights |
| `pycardinal.summarize` | Available: dense/sparse row and column reductions plus grouped feature/pixel summaries |
| `pycardinal.plotting` | Available baseline: spectra, ion-image, and model plots |
| `pycardinal.roi` | Available: deterministic polygon/point ROI masks and categorical mask factors |
| `pycardinal.stats` | Available baseline: PCA, NMF, PLS/OPLS, spatial projections and clustering, DGMM, means tests, cross-validation, and feature ranking |

The repository-level `PYTHON_API_REFERENCE.md` contains the detailed planned
signatures, parameters, and mappings to the existing R documentation. Entries
there are a target specification unless also listed above as available.
