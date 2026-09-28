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

The published distribution name is `python-cardinal`; the Python import name
remains `pycardinal`.

The default test suite uses deterministic simulated data and does not require
private datasets. The completed
migration history is in [PYTHON_MIGRATION.md](PYTHON_MIGRATION.md).

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
the Cardinal mass-spectrometry imaging project. The original R implementation
and its license are preserved on the `devel` branch. Please acknowledge the
original Cardinal authors and publication when using work derived from Cardinal.

Suggested citation for the original Cardinal work:

> Bemis, K. A., Foell, M. C., Guo, D., Lakkimsetty, S. S., and Vitek, O.
> “Cardinal v.3: a versatile open-source software for mass spectrometry imaging
> analysis.” *Nature Methods* 20, 1883–1886 (2023).
> [doi:10.1038/s41592-023-02070-z](https://doi.org/10.1038/s41592-023-02070-z)

