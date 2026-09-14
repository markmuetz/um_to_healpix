"""Checks on a written healpix zarr store: coverage, structure, NaNs, value ranges, level consistency.

Used by the `check_store` rule in remakefile_coarsen.py (which writes one JSON report per store) and by
scripts/compare_stores.py. Kept free of remake/SLURM specifics so it can be unit tested offline.

Each check returns a list of failure strings; an empty list means the check passed.
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

# Physically plausible ranges (vmin, vmax); None means unbounded. Mirrors output_tests/datasets.py.
RANGE_CHECKS = {
    'tas': (150, 400),  # K
    'pr': (0, None),  # kg m-2 s-1, non-negative
    'rlut': (0, None),  # W m-2, non-negative
    'clw': (0, None),  # kg kg-1, non-negative
}
# Permits tiny negative values from floating-point rounding during regridding (observed ~1e-18).
RANGE_CHECK_ATOL = 1e-10
# Global sims should have less than this fraction of NaNs per variable at any zoom.
MAX_NAN_FRACTION_GLOBAL = 0.50
# Land-only variables are NaN over ocean, so their NaN fraction tends to the ocean fraction (~71%) as zoom
# increases and coastlines are resolved (mrsol: 53% at z4 -> 68% at z7). Judge them against a land-aware limit.
MAX_NAN_FRACTION_BY_VAR = {
    'mrsol': 0.80,  # soil moisture: land only
}
# Tolerance when checking that a coarsened field equals the mean of its 4 children: |a-b| <= atol + rtol*|b|,
# with atol a fraction of the field's own magnitude. Fields that cross zero (wa, winds) have cells whose value is
# ~0, where a relative test alone is meaningless: float32 summation order in the coarsening then shows up as a
# large relative difference on a physically negligible number.
CONSISTENCY_RTOL = 1e-4
CONSISTENCY_ATOL_FRAC = 1e-5
# Reading whole fields to compare levels is expensive at high zoom; only do it at or below this zoom.
MAX_CONSISTENCY_ZOOM = 7


def written_time_steps(fs, url, var, ntime, time_chunk=1):
    """Time steps of var for which every chunk exists in the store.

    Zarr keys index *chunks*: below z9 one chunk spans many time steps (time_chunk), so a chunk index must be
    expanded to the steps it covers or coverage is under-reported by that factor.
    """
    counts = defaultdict(int)
    for key in fs.ls(f'{url[5:]}/{var}', detail=False):
        name = key.rsplit('/', 1)[-1]
        if re.fullmatch(r'\d+(\.\d+)+', name):
            counts[int(name.split('.')[0])] += 1
    if not counts:
        return []
    full = max(counts.values())
    steps = []
    for chunk, n in counts.items():
        if n == full:
            steps.extend(range(chunk * time_chunk, min((chunk + 1) * time_chunk, ntime)))
    return sorted(steps)


def check_coverage(fs, url, ds):
    """Every variable has every time step written, with no gaps."""
    failures = []
    ntime = ds.sizes['time']
    for var in sorted(v for v in ds.data_vars if 'time' in ds[v].dims):
        chunks = ds[var].encoding.get('chunks') or (1,)
        steps = written_time_steps(fs, url, var, ntime, time_chunk=chunks[0])
        if not steps:
            failures.append(f'{var}: nothing written')
            continue
        missing = ntime - len(steps)
        gaps = steps[-1] - steps[0] + 1 - len(steps)
        if missing:
            failures.append(f'{var}: {len(steps)}/{ntime} time steps written '
                            f'(first {steps[0]}, last {steps[-1]}, {gaps} internal gaps)')
    return failures


def check_structure(ds, expected_time, expected_ncell, expected_vars=None):
    """Coords and dimensions are the ones the config asks for."""
    failures = []
    if 'time' not in ds.dims:
        failures.append('no time dimension')
    elif ds.sizes['time'] != len(expected_time):
        failures.append(f'time has {ds.sizes["time"]} steps, expected {len(expected_time)}')
    elif not pd.DatetimeIndex(ds.time.values).equals(pd.DatetimeIndex(expected_time)):
        failures.append('time coordinate values differ from the config time index')
    cell_dim = 'cell' if 'cell' in ds.dims else 'healpix_index'
    if cell_dim not in ds.dims:
        failures.append('no cell/healpix_index dimension')
    elif ds.sizes[cell_dim] != expected_ncell:
        failures.append(f'{cell_dim} has {ds.sizes[cell_dim]}, expected {expected_ncell}')
    if expected_vars is not None:
        missing = set(expected_vars) - set(ds.data_vars)
        if missing:
            failures.append(f'missing variables: {sorted(missing)}')
    return failures


def _sample(da, time_idx):
    """One time step of a variable, with any extra dimension (pressure, depth) kept."""
    return da.isel(time=time_idx).values


def check_nans_and_ranges(ds, time_idx, max_nan_fraction=MAX_NAN_FRACTION_GLOBAL):
    """NaN fraction and physically plausible values, per variable and per extra-dimension level.

    mrsol is hourly but has a depth dimension, so levels are handled generically rather than by group.
    """
    failures = []
    for var in sorted(v for v in ds.data_vars if 'time' in ds[v].dims):
        values = _sample(ds[var], time_idx)
        extra_dims = [d for d in ds[var].isel(time=0).dims if d not in ('cell', 'healpix_index')]
        levels = values.reshape(-1, values.shape[-1]) if extra_dims else values[np.newaxis, :]
        limit = MAX_NAN_FRACTION_BY_VAR.get(var, max_nan_fraction)
        for i, level in enumerate(levels):
            label = f'{var}[{extra_dims[0]}={i}]' if extra_dims else var
            nan_fraction = float(np.isnan(level).mean())
            if nan_fraction > limit:
                failures.append(f'{label}: {nan_fraction:.1%} NaN (limit {limit:.0%})')
            if nan_fraction == 1.0:
                continue
            vmin, vmax = RANGE_CHECKS.get(var, (None, None))
            finite = level[~np.isnan(level)]
            if vmin is not None and finite.min() < vmin - RANGE_CHECK_ATOL:
                failures.append(f'{label}: min {finite.min():.6g} below {vmin}')
            if vmax is not None and finite.max() > vmax + RANGE_CHECK_ATOL:
                failures.append(f'{label}: max {finite.max():.6g} above {vmax}')
    return failures


def check_level_consistency(ds_coarse, ds_fine, time_idx, rtol=CONSISTENCY_RTOL,
                            atol_frac=CONSISTENCY_ATOL_FRAC, vars_to_check=None):
    """A coarsened field equals the mean of its 4 children (HEALPix nested ordering)."""
    failures = []
    variables = vars_to_check or sorted(set(ds_coarse.data_vars) & set(ds_fine.data_vars))
    for var in variables:
        if 'time' not in ds_coarse[var].dims:
            continue
        coarse = _sample(ds_coarse[var], time_idx).astype(np.float64)
        fine = _sample(ds_fine[var], time_idx).astype(np.float64)
        if fine.shape[-1] != 4 * coarse.shape[-1]:
            failures.append(f'{var}: fine has {fine.shape[-1]} cells, expected 4x{coarse.shape[-1]}')
            continue
        with np.errstate(invalid='ignore'):
            expected = np.nanmean(fine.reshape(*fine.shape[:-1], coarse.shape[-1], 4), axis=-1)
        both = ~np.isnan(expected) & ~np.isnan(coarse)
        if not both.any():
            continue
        scale = float(np.nanmax(np.abs(expected))) if np.isfinite(expected).any() else 0.0
        atol = atol_frac * scale
        diff = np.abs(coarse[both] - expected[both])
        bad = diff > atol + rtol * np.abs(expected[both])
        n_bad = int(bad.sum())
        if n_bad:
            worst = diff[bad].max()
            failures.append(f'{var}: {n_bad}/{both.sum()} cells differ from the mean of their 4 children '
                            f'(max abs {worst:.3g} > atol {atol:.3g} + rtol {rtol:g}*|value|)')
    return failures


def open_store(url, s3, **kwargs):
    """Open a zarr store from an s3 URL (or a local path, for tests)."""
    import s3fs
    if str(url).startswith('s3://'):
        return xr.open_zarr(s3fs.S3Map(root=url, s3=s3, check=False), consolidated=True, **kwargs)
    return xr.open_zarr(url, consolidated=True, **kwargs)
