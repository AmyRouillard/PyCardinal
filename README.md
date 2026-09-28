# PyCardinal and Cardinal

## Python package

**pycardinal** is a direct Python port of the original R/Bioconductor Cardinal
package for mass-spectrometry imaging data. It ports Cardinal's data containers,
imzML I/O, spectral processing, feature and spatial utilities, summary
statistics, statistical models, plotting, and ROI helpers into a Python-native
API. Some statistical methods remain practical baseline implementations and are
documented where they are not yet numerically identical to R Cardinal.

Install the development environment from the repository root:

```sh
python -m pip install -e ".[dev,docs]"
```

The default test suite uses deterministic simulated data and does not require
private datasets. Exact R numerical parity and release hardening are tracked in
the [deployment readiness plan](DEPLOYMENT_READINESS_PLAN.md); the completed
migration history is in [PYTHON_MIGRATION_PLAN.md](PYTHON_MIGRATION_PLAN.md).

### Quickstart

```python
from pycardinal import PCA, plot_image, process, read_imzml

image = read_imzml("sample.imzML")
image = process(image)
model = PCA(image, ncomp=2)
plot_image(image, i=0)
print(model.sdev)
```

For a self-contained example without an input file:

```python
from pycardinal import PCA, plot_image, simulate_image

image = simulate_image(preset=1, dim=(10, 10), npeaks=12, random_state=7)
model = PCA(image, ncomp=2)
plot_image(image, i=0)
```

See the [Python documentation](docs/index.md), [generated API](docs/generated-api.md),
and [parameter reference](PYTHON_API_REFERENCE.md) for details. The maintained
fork is hosted at [github.com/AmyRouillard/PyCardinal](https://github.com/AmyRouillard/PyCardinal).

### Attribution

PyCardinal is maintained by Amy Rouillard as a direct Python port and fork of
the Cardinal mass-spectrometry imaging project. The original R implementation and
its license are preserved under [`legacy-r/`](legacy-r/). Please acknowledge the
original Cardinal authors and publication when using work derived from Cardinal;
the citation details are preserved in [legacy-r/inst/CITATION](legacy-r/inst/CITATION).

Suggested citation for the original Cardinal work:

> Bemis, K. A., Foell, M. C., Guo, D., Lakkimsetty, S. S., and Vitek, O.
> “Cardinal v.3: a versatile open-source software for mass spectrometry imaging
> analysis.” *Nature Methods* 20, 1883–1886 (2023).
> [doi:10.1038/s41592-023-02070-z](https://doi.org/10.1038/s41592-023-02070-z)

## Cardinal R package

*Cardinal* provides an R/Bioconductor interface for manipulating mass
spectrometry imaging datasets. The original R implementation remains the
behavior and numerical reference for this migration, archived under
[`legacy-r/`](legacy-r/). The Python package at the repository root is the
maintained implementation.

## User Installation

### Bioconductor Release

*Cardinal* can be installed via the *BiocManager* package.

This is the **recommended** installation method.

```{r install, eval=FALSE}
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("Cardinal")
```

The same function can be used to update *Cardinal* and other Bioconductor packages.

Once installed, *Cardinal* can be loaded with `library()`:

```{r library, eval=FALSE}
library(Cardinal)
```

### Github Release

*Cardinal* can also be installed via the *remotes* package.

```{r install, eval=FALSE}
if (!require("remotes", quietly = TRUE))
    install.packages("remotes")

remotes::install_github("kuwisdelu/Cardinal", ref=remotes::github_release())
```

Previous releases can be installed by specifying the exact version.

```{r library, eval=FALSE}
remotes::install_github("kuwisdelu/Cardinal@v3.6.2")
```

## Developer Installation

### Bioconductor Devel

The Bioconductor development version of *Cardinal* can also be installed via the *BiocManager* package.

```{r install, eval=FALSE}
BiocManager::install("Cardinal", version="devel")
```

This version is **unstable** and should not be used for critical work. However, it is typically more stable than Github devel.

This version should *typically* pass `R CMD check` without errors.

### Github Devel

The most cutting edge version of *Cardinal* can be installed from Github via the *remotes* package.

```{r install, eval=FALSE}
if (!require("remotes", quietly = TRUE))
    install.packages("remotes")

remotes::install_github("kuwisdelu/Cardinal")
```

This version is **unstable** and only recommended for developers. It should not be used for critical work.


