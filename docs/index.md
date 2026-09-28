# pycardinal

Python mass spectrometry imaging tools directly ported from the R/Bioconductor
Cardinal package.

## Project status

The Python package is a direct Python port of Cardinal and has completed Phases
0–11 at baseline level: core data,
imzML I/O, spectral processing, feature/pixel and spatial utilities, summaries,
statistical models, plotting, ROI helpers, and simulation. Synthetic tests cover these available
slices, including imzML round trips; no private dataset is required. Exact R
numerical parity remains open for optional comparison in an R/Cardinal
environment. The original R code is preserved under `legacy-r/` as the
behavior and numerical reference; the Python package is the maintained root
project.

## Start here

- [Installation and synthetic-data testing](installation.md)
- [API reference and current module map](api-reference.md)
- [Development and documentation conventions](development.md)

The repository-level `PYTHON_MIGRATION_PLAN.md` tracks implementation phases;
`PYTHON_API_REFERENCE.md` is the detailed target function and parameter index.
