# Test Suite

## Overview

Three layers of automated tests alongside the existing `output_tests` system tests.

```bash
pytest tests/          # unit + integration + e2e
pytest output_tests/   # remote system tests (separate, needs S3 access)
```

See `tests/README.md` for a quick-reference command cheat sheet.

---

## Existing tests: `output_tests/`

System-level smoke tests against published remote zarr stores. Run manually
after a processing run completes, not in CI.

**What they test:** catalog discovery, lazy dataset opening, single-timestep
loading, NaN fraction checks, physical value ranges, and plot generation for
all 12 HK26 simulations × 11 zooms × 2 frequencies.

```bash
cd output_tests
pytest test_hk26_data.py                            # data checks (needs S3)
pytest test_plots.py -m plots --plot-dir /tmp/figs  # plot generation
```

---

## New test suite

### Known version constraint

**easygems 0.0.14** is required (pinned in `pyproject.toml`). Versions 0.1.x
introduced a stereographic transform in `compute_weights_delaunay` that
produces NaN for exact ±90° latitude inputs, crashing `scipy.spatial.Delaunay`.

### Directory structure

```
tests/
├── README.md
├── conftest.py               # make_test_config() helper, local_store_factory
├── unit/                     # pure in-memory, no filesystem
├── integration/              # synthetic grid + local zarr, session-scoped weights
└── e2e/                      # full three-stage pipeline on local filesystem
output_tests/                 # unchanged — remote system tests
```

---

## Code changes

Six targeted changes to production code to enable testing without S3 or real
`.pp` files.

1. **Move `.s3cfg` read out of module scope** (`um_process_tasks.py`) — was
   crashing on import if `~/.s3cfg` absent; moved into `get_jasmin_s3()`.

2. **Storage backend injection** — `UMProcessTasks.__init__` accepts an optional
   `store_factory` callable. Default wraps `s3fs.S3Map`; tests pass
   `lambda url: url` (zarr accepts plain path strings).

3. **`store=` parameter on `healpix_da_to_zarr`** — accepts a pre-created store;
   falls back to `_default_store_factory(url)` when `None`.

4. **Extract `_compute_coarsened`** from `coarsen_healpix_zarr_region`
   (`healpix_coarsen.py`) — pure computation returning `xr.Dataset`, no zarr
   writes. Also changed `drop_vars('crs')` → `drop_vars('crs', errors='ignore')`
   for robustness.

5. **`cubes=` parameter on `regrid()`** — bypasses `iris.load` when a
   `CubeList` is supplied directly.

6. **`cubes=` parameter on `create_empty_zarr_stores()`** — same pattern;
   also made `air_pressure`/`geopotential_height` extraction optional so
   2D-only configs don't require those cubes.

---

## Unit tests

No network, no filesystem, no real cubes. All synthetic numpy/xarray/iris data.

- `test_latlon_to_healpix.py` — cyclic point, extent, regional cell indices,
  regridder init validation
- `test_healpix_coarsen.py` — `nan_weight`, `map_regional_to_global`,
  `map_global_to_regional`
- `test_cube_to_da_mapping.py` — `MapItem`, `MultiMapItem`, extractor trimming
  and error handling
- `test_util.py` — `find_halfpast_time`, `weights_filename`, `invert_cube_sign`,
  `make_percentage`, `has_dimensions`

---

## Integration tests

Use a 24×12 synthetic grid (0–345° lon, ±82.5° lat) at zoom=2. Weights are
generated once per test session by a session-scoped fixture.

**Grid note:** exact ±90° latitudes crash the Delaunay triangulation in
easygems 0.0.14 via divide-by-zero in the stereographic projection.
The ±82.5° range avoids this.

- `test_regrid_pipeline.py` — output shape, no all-NaN, mean preservation,
  name preservation; tested for 2D and 3D (time) inputs
- `test_coarsen_pipeline.py` — tests `_compute_coarsened` directly: cell count
  reduction (768→192), time preservation, metadata attrs, mean preservation,
  value range

---

## End-to-end tests

Full three-stage pipeline on synthetic data writing to a local temp directory.
No `.pp` files, no S3. `_gen_orog_land_sea` is patched with synthetic datasets.

The `e2e_pipeline` module-scoped fixture runs:
1. `create_empty_zarr_stores` — pre-allocates stores at zoom 0, 1, 2
2. `regrid` — fills zoom=2 store
3. `coarsen_healpix_zarr_region` loop — fills zoom=1 and zoom=0

Tests verify zarr stores exist at every zoom with the right variables
(`tas`, `orog`, `sftlf`), correct cell counts (`12·4^z`), non-NaN regridded
values in a plausible temperature range, and mean conservation through
the coarsen steps.

---

## `pyproject.toml` additions

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
markers = [
    "slow: marks tests as slow (weight generation, zarr writes)",
    "e2e: end-to-end tests requiring synthetic pipeline setup",
]

[project.optional-dependencies]
test = ["pytest"]
```
