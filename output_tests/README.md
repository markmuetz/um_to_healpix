# HK26 output tests

Pytest-based tests for validating HK26 datasets. Run from the `um_to_healpix/` root or directly from this directory.

## Basic usage

```bash
# Run all data quality tests (default catalog, no plots)
pytest output_tests/

# Same, but run from this directory
pytest
```

## Catalog options

By default tests hit the main hackathon catalog and select the `UK` sub-catalog:

```bash
# Use a different catalog branch (no sub-catalog key needed)
pytest --catalog-url https://raw.githubusercontent.com/digital-earths-global-hackathon/catalog/refs/heads/uk_hk26_sims2/UK/main.yaml --catalog-key ''

# Use a local catalog file
pytest --catalog-url /path/to/catalog.yaml --catalog-key ''
```

## Plot tests

Plot tests are opt-in via the `plots` marker. They save figures to `output_tests/figures/` by default.

```bash
# Run plot tests
pytest -m plots

# Run plot tests to a custom directory
pytest -m plots --plot-dir /path/to/figures

# Run everything (data tests + plot tests)
pytest -m 'plots or not plots'
```

Output is organised as `{plot-dir}/{sim}/{freq}/`:
- `all_fields.z{zoom}.png` — all variables as healpix maps at a single timestep
- `pr_zonal_mean.png` — precipitation zonal mean across zooms 0–5
- `pr_timeseries.png` — global mean precipitation timeseries across zooms 0–5
- `clw_pressure_profile.z{zoom}.png` — cloud liquid water vertical profile

## Filtering tests

Use `-k` to match against test IDs, which have the form `{sim}-z{zoom}-{freq}`:

```bash
# One sim only
pytest -k glm_n2560_RAL3p3_tuned

# One sim, one zoom, one freq
pytest -k "glm_n2560_RAL3p3_tuned and z3 and PT1H"

# All zoom=0 tests
pytest -k z0

# All PT3H tests
pytest -k PT3H

# Smoke test plots: one sim, one zoom
pytest -m plots -k "glm_n2560_RAL3p3_tuned and z3"
```

## Capturing output

```bash
# Write output to file and show in terminal simultaneously
pytest -v 2>&1 | tee pytest_output.txt

# Write to file only (silent)
pytest -v > pytest_output.txt 2>&1
```

## Common combinations

```bash
# Full run against the branch catalog, capturing output
pytest -v \
  --catalog-url https://raw.githubusercontent.com/digital-earths-global-hackathon/catalog/refs/heads/uk_hk26_sims2/UK/main.yaml \
  --catalog-key '' \
  2>&1 | tee pytest_output.txt

# Full run including plots
pytest -v -m 'plots or not plots' --plot-dir figures 2>&1 | tee pytest_output.txt
```
