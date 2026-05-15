# Test Suite Design

## Overview

This document describes the planned test suite for `um_to_healpix`. The goal is to add unit,
integration, and end-to-end tests alongside the existing `output_tests` system tests.

---

## Existing tests: `output_tests/`

The `output_tests/` directory contains system-level smoke tests that run against published,
remote data. They are **not** part of the new test suite and are not intended to run in CI.

### What they test

- **Catalog discovery** (`test_hk26_data.py`): verifies that all expected simulations appear in
  the intake catalog and no unregistered `_hk26` simulations exist.
- **Dataset opens** (`test_hk26_data.py`): for every `(sim, zoom, freq)` combination (12 sims ×
  11 zooms × 2 freqs = 264 cases), checks that the remote zarr store opens lazily and has the
  expected dimensions.
- **Single timestep loads** (`test_hk26_data.py`): fetches one timestep per combination and
  verifies that it loads without error.
- **NaN checks** (`test_hk26_data.py`): checks that global sims have < 50% NaN per variable;
  LAM sims are skipped at zoom ≤ 2.
- **Value range checks** (`test_hk26_data.py`): checks physical plausibility for `tas`, `pr`,
  `rlut`, `clw`.
- **No NaNs at zoom=0** (`test_hk26_data.py`): full time-series check at zoom=0 (emits
  warnings rather than failures).
- **Plot outputs** (`test_plots.py`): renders and saves PNG plots for all fields, zonal mean
  precipitation, precipitation time series, and cloud-water pressure profiles. Marked with
  `pytest.mark.plots`.

### How they are run

```bash
cd output_tests
pytest test_hk26_data.py                  # data checks (needs network/S3 access)
pytest test_plots.py -m plots             # plot generation
pytest --catalog-url <url> --plot-dir /tmp/plots  # custom options
```

These tests are run manually after a processing run completes, not in CI.

---

## New test suite

### Framework

**pytest** — stays as-is. No additional libraries needed:

- `unittest.mock` (stdlib) covers all mocking needs; `pytest-mock` is not required.
- Storage injection (code change 2) eliminates the need for `moto` or S3 mocking.

`hypothesis` is not added at this stage.

### Directory structure

```
tests/
├── conftest.py               # shared fixtures and make_test_config()
├── unit/
│   ├── test_latlon_to_healpix.py
│   ├── test_healpix_coarsen.py
│   ├── test_cube_to_da_mapping.py
│   └── test_util.py
├── integration/
│   ├── conftest.py           # weight-generation fixture (session-scoped, slow)
│   ├── test_regrid_pipeline.py
│   └── test_coarsen_pipeline.py
└── e2e/
    └── test_full_pipeline.py
output_tests/                 # unchanged — remote system tests
```

`pytest.ini` / `pyproject.toml` configuration will mark tests so that:

```bash
pytest tests/                              # all new tests (unit + integration + e2e)
pytest tests/ -m "not slow"               # unit tests only
pytest tests/unit/                         # unit tests only (explicit)
pytest output_tests/                       # existing system tests (separate)
```

---

## Required code changes

Four targeted changes to production code; no structural redesign.

### 1. Remove module-level `.s3cfg` read (`um_process_tasks.py`)

**Current** (line 42, runs at import time, crashes if file absent):
```python
s3cfg = dict([l.split(' = ') for l in (Path.home() / '.s3cfg').read_text().split('\n') if l])
```

**Fix**: move into `get_jasmin_s3()` so it is only evaluated when actually connecting to S3.

### 2. Inject storage backend into `UMProcessTasks`

Add a `store_factory` callable parameter:

```python
class UMProcessTasks:
    def __init__(self, config, shared_metadata, store_factory=None):
        ...
        # Defaults to the real JASMIN S3 factory
        self._store_factory = store_factory or _default_s3_store_factory
```

`_default_s3_store_factory(url)` wraps the existing `s3fs.S3Map(root=url, s3=get_jasmin_s3())`.
Tests pass a local-filesystem factory: `lambda url: url`  (zarr accepts a plain path string).

### 3. Extract compute from IO in `coarsen_healpix_zarr_region`

Split the function into:
- `_compute_coarsened(src_ds, tgt_zoom, dim, start_idx, end_idx, chunks, regional)` → `xr.Dataset`
- `coarsen_healpix_zarr_region(...)` calls `_compute_coarsened` then writes; signature unchanged

This makes the coarsening computation independently testable without zarr.

### 4. Add optional `cubes` parameter to `UMProcessTasks.regrid()`

```python
def regrid(self, task, cubes=None):
    if cubes is None:
        cubes = iris.load(task['inpaths'])
    ...
```

Allows integration tests to pass in a synthetic in-memory `CubeList` without touching the
filesystem.

---

## Test helper: `make_test_config()`

Defined in `tests/conftest.py`. Assembles a minimal valid config dict so individual tests
do not need to construct the full nested structure by hand.

```python
def make_test_config(tmp_path, *, zoom=2, regional=False, **overrides):
    """Return a minimal processing config dict suitable for tests."""
    ...
```

---

## Unit tests

No network, no filesystem, no real cubes. Use synthetic `np.ndarray` / `xr.DataArray` inputs.

### `tests/unit/test_latlon_to_healpix.py`

| Test | What it checks |
|---|---|
| `test_add_cyclic_point_adds_one_column` | Output longitude dim is `len(input) + 1` |
| `test_add_cyclic_point_wrap_value_matches` | Wrapped column data equals column at lon=0 |
| `test_get_extent_returns_correct_bounds` | Known DataArray → expected `(minlon, maxlon, minlat, maxlat)` |
| `test_get_regional_cell_idx_returns_cells` | Small box at low zoom → non-empty cell array |
| `test_get_regional_cell_idx_wrapped_lon` | Extent with `maxlon > 360` (Africa domain) → cells returned |
| `test_regridder_raises_on_bad_method` | `ValueError` for unknown method string |
| `test_regridder_accepts_each_valid_method` | No exception for `easygems_delaunay`, `easygems_delaunay_parallel`, `earth2grid` (init only, no regrid call) |

### `tests/unit/test_healpix_coarsen.py`

| Test | What it checks |
|---|---|
| `test_nan_weight_all_valid` | 4 valid values → weight = 1.0 |
| `test_nan_weight_half_nan` | 2 NaN + 2 valid → weight = 0.5 |
| `test_nan_weight_all_nan` | 4 NaN → weight = 0.0 |
| `test_map_regional_to_global_full_shape` | Output `healpix_index` has global size; regional values placed correctly; rest NaN |
| `test_map_regional_to_global_weights` | Works for the `weights` variable (1D, no time dim) |
| `test_map_global_to_regional_shape` | Output has only target-region cells |

### `tests/unit/test_cube_to_da_mapping.py`

| Test | What it checks |
|---|---|
| `test_map_item_repr` | `repr()` does not raise |
| `test_multi_map_item_repr` | `repr()` does not raise |
| `test_multi_map_item_extra_attrs_default` | `extra_attrs` defaults to `{}` when not supplied |
| `test_extract_cubes_trims_length_13_to_12` | Cube with `shape[0] == 13` is sliced to `[1:]` |

### `tests/unit/test_util.py`

| Test | What it checks |
|---|---|
| `test_find_halfpast_time_detects_half_past` | DataArray with `:30` minutes returns coord name |
| `test_find_halfpast_time_returns_none` | DataArray with `:00` minutes returns `None` |
| `test_weights_filename_contains_zoom` | Filename string contains `hpz<zoom>` |
| `test_weights_filename_contains_lon_lat_info` | Filename contains lon/lat range and grid size |
| `test_invert_cube_sign` | Output data == `-1 × input data` |
| `test_make_percentage` | Output data == `100 × input data` |
| `test_has_dimensions_matches_correct_cube` | Constraint extracts cube with right dim tuple |
| `test_has_dimensions_rejects_wrong_dims` | Constraint returns empty list for wrong dims |

---

## Integration tests

Use a small synthetic grid (e.g. 20 lon × 10 lat) at zoom=2. A session-scoped fixture generates
weights once for the whole test session (slow but only once).

### `tests/integration/conftest.py`

- `tiny_da` (session): `xr.DataArray` with lon 0–360 (20 pts), lat −90–90 (10 pts), 3 time steps,
  values = `sin(lat) + 0.1 * noise`
- `tiny_weights` (session, `tmp_path_factory`): calls `gen_weights()` on `tiny_da` at zoom=2;
  result cached for the session

### `tests/integration/test_regrid_pipeline.py`

| Test | What it checks |
|---|---|
| `test_regrid_output_shape` | `da_hp.dims == ('time', 'healpix_index')` and size matches `12 * 4**zoom` |
| `test_regrid_no_nans_global` | No NaN values in global (non-regional) regrid output |
| `test_regrid_global_mean_approximately_preserved` | `abs(da.mean() - da_hp.mean()) / da.mean() < 0.05` |
| `test_regrid_cyclic_vs_no_cyclic_differ` | Results are not identical when `add_cyclic` is toggled |
| `test_regrid_preserves_name` | Output DataArray `name` attribute matches input |

### `tests/integration/test_coarsen_pipeline.py`

Uses a synthetic zarr store written to `tmp_path` (no S3).

| Test | What it checks |
|---|---|
| `test_coarsen_output_ncells` | Output has `12 * 4**tgt_zoom` cells (4× fewer than source) |
| `test_coarsen_output_mean_close_to_input` | Spatial mean preserved to within 1% |
| `test_coarsen_all_nan_does_not_raise` | An all-NaN time slice triggers `logger.error`, not an exception |

---

## End-to-end tests

A minimal three-stage pipeline on fully synthetic in-memory data, writing to a local temporary
directory (no S3, no `.pp` files). The `cubes=` kwarg (code change 4) allows `regrid()` to
accept a hand-crafted `CubeList`.

### `tests/e2e/test_full_pipeline.py`

| Test | What it checks |
|---|---|
| `test_create_zarr_stores` | `create_empty_zarr_stores()` produces zarr directories for each zoom |
| `test_regrid_populates_zarr` | After `regrid()`, zarr store contains non-NaN data for at least one variable |
| `test_coarsen_produces_lower_zooms` | After `coarsen_healpix_region()`, lower-zoom zarr stores exist and have correct cell count |
| `test_full_pipeline_end_to_end` | All three stages run sequentially; final zoom=0 store has expected variables and time length |

---

## `pyproject.toml` additions

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (weight generation, zarr writes)",
    "e2e: end-to-end tests requiring synthetic pipeline setup",
]

[project.optional-dependencies]
test = [
    "pytest",
]
```

---

## Implementation order

1. Code changes (4 targeted refactors)
2. Unit tests
3. Integration tests (after verifying unit tests pass)
4. End-to-end tests
