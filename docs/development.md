# Development

## Project structure

- `src/pycardinal/` contains the installable Python package.
- `tests/` contains pytest checks built from deterministic simulated spectra
  and temporary imzML files. No external MSI dataset is required.
- `docs/` contains the MkDocs site.
- `docs/api-reference.md` is the hand-written behavioral guide;
  `docs/generated-api.md` is generated from public Python docstrings.
- The original Cardinal authors and publication remain the migration attribution
  and numerical reference.

## Checks

Run these from the repository root after a code or documentation change:

```sh
python -m pytest
python -m ruff check src tests
python -m mypy src/pycardinal
python -m mkdocs build --strict
```

If a future manual integration test uses a large private imzML file, keep it
outside the default pytest suite and document its local setup separately.

## API documentation conventions

Every public class and function should have a NumPy-style docstring with a
summary, parameter descriptions and units, return values, raised exceptions,
and a small executable example where practical. Document whether processing
is deferred or eager, the orientation of data axes, and any relevant m/z or
spatial units. Update the API reference and relevant guide in the same change
that adds or changes public behavior.
