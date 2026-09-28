# Installation and Testing

## Requirements

- Python 3.14 or newer.
- A virtual environment is recommended.

## Install for development

From the repository root, create and activate an environment, then install the
package with development and documentation tools:

=== "Windows PowerShell"

    ```powershell
    py -3.14 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev,docs]"
    ```

=== "macOS / Linux"

    ```sh
    python3.14 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev,docs]"
    ```

The `-e` option installs the working tree in editable mode. `dev` includes
pytest, Ruff, mypy, and Hypothesis. `docs` includes MkDocs Material and
mkdocstrings.

## Build distributions

The optional `packaging` extra installs the release checks:

```sh
python -m pip install -e ".[packaging]"
python -m build
python -m twine check dist/*
```

This produces a source archive and wheel in `dist/`. Install the wheel directly
for a non-editable local validation:

```sh
python -m pip install --force-reinstall dist/pycardinal-*.whl
```

The package requires Python `>=3.14`. Python 3.14.2 is supported, along with
source and wheel artifact installation.

## Tests

Run the fast suite with:

```sh
python -m pytest
```

The test suite is self-contained: it creates small simulated spectra and
temporary imzML/ibd pairs, so no private or downloaded mass-spectrometry data is
required. Your own imzML files can still be opened by `read_imzml()` for manual
analysis; they are not automatically read by pytest.

## Documentation site

Preview the docs locally with:

```sh
python -m mkdocs serve
```

Build the site and treat documentation warnings as errors with:

```sh
python -m mkdocs build --strict
```
