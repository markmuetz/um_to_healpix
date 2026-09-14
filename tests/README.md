# Tests

## Quick start

```bash
# All tests
pytest tests/

# Unit tests only (fast, ~4s)
pytest tests/unit/

# Integration tests (weight generation, ~3s)
pytest tests/integration/

# End-to-end tests (full pipeline, ~2s)
pytest tests/e2e/
```

## Filtering by marker

```bash
# Skip slow tests (weight generation, zarr writes)
pytest tests/ -m "not slow"

# Run only e2e tests
pytest tests/ -m e2e
```

## Useful flags

```bash
# Verbose output (show each test name)
pytest tests/ -v

# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l

# Run tests matching a keyword
pytest tests/ -k "coarsen"
```

## Existing system tests (remote data)

The `output_tests/` directory contains smoke tests against the published
zarr stores and requires network access to the JASMIN S3 object store.
These are run manually after a processing run, not in CI.

```bash
cd output_tests
pytest test_hk26_data.py                         # data quality checks
pytest test_plots.py -m plots --plot-dir /tmp/figs  # plot generation
```

## Dependencies

All test dependencies are included in the main conda environment.
No extra packages needed beyond what is already installed.

> **Note:** `easygems==0.0.14` is required (pinned in `pyproject.toml`).
> Versions 0.1.x introduced a regression in `compute_weights_delaunay`
> that causes a crash for grids containing exact ±90° latitudes.
