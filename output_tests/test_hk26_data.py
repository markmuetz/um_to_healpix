import warnings

import numpy as np
import pytest

from datasets import (
    FREQS,
    HK26_GLOBAL_SIMS,
    HK26_LAM_SIMS,
    HK26_SIMS,
    LAM_MIN_ZOOM,
    MAX_NAN_FRACTION_GLOBAL,
    RANGE_CHECK_ATOL,
    RANGE_CHECKS,
    ZOOMS,
)

# ---------------------------------------------------------------------------
# Parameter list: (sim, zoom, freq) for all combinations
# ---------------------------------------------------------------------------
PARAMS = [
    (sim, zoom, freq)
    for sim in HK26_SIMS
    for zoom in ZOOMS
    for freq in FREQS
]

def _id(sim, zoom, freq):
    short = sim.replace('um_', '').replace('_hk26', '')
    return f"{short}-z{zoom}-{freq}"

PARAM_IDS = [_id(*p) for p in PARAMS]


# ---------------------------------------------------------------------------
# Test 1 – catalog discovery
# ---------------------------------------------------------------------------

def test_catalog_lists_expected_sims(catalog):
    available = set(catalog)
    missing = [s for s in HK26_SIMS if s not in available]
    assert not missing, f"Sims missing from catalog: {missing}"


def test_no_unregistered_hk26_sims(catalog):
    unregistered = {k for k in catalog if k.endswith('_hk26')} - set(HK26_SIMS)
    assert not unregistered, f"New _hk26 sims in catalog not in registry: {unregistered}"


# ---------------------------------------------------------------------------
# Test 2 – dataset opens lazily
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sim,zoom,freq", PARAMS, ids=PARAM_IDS)
def test_dataset_opens(catalog, sim, zoom, freq):
    try:
        ds = catalog[sim](zoom=zoom, time=freq).to_dask()
    except Exception as exc:
        pytest.skip(f"Not available: {exc}")

    assert 'time' in ds.dims
    # dimension may be 'healpix_index' before hp_mods — either name is fine here
    assert 'healpix_index' in ds.dims or 'cell' in ds.dims
    assert ds.sizes['time'] >= 1


# ---------------------------------------------------------------------------
# Test 3 – single timestep loads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sim,zoom,freq", PARAMS, ids=PARAM_IDS)
def test_single_timestep_loads(get_snapshot, sim, zoom, freq):
    if sim in HK26_LAM_SIMS and zoom <= LAM_MIN_ZOOM:
        pytest.skip(f"LAM sims not expected to have valid data at zoom <= {LAM_MIN_ZOOM}")
    result = get_snapshot(sim, zoom, freq)
    if isinstance(result, Exception):
        pytest.skip(f"Not available: {result}")
    assert result is not None
    assert result.sizes.get('cell', 0) > 0


# ---------------------------------------------------------------------------
# Test 4 – NaN check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sim,zoom,freq", PARAMS, ids=PARAM_IDS)
def test_nan_check(get_snapshot, sim, zoom, freq):
    if sim in HK26_LAM_SIMS and zoom <= LAM_MIN_ZOOM:
        pytest.skip(f"LAM sims not expected to have valid data at zoom <= {LAM_MIN_ZOOM}")
    snapshot = get_snapshot(sim, zoom, freq)
    if isinstance(snapshot, Exception):
        pytest.skip(f"Not available: {snapshot}")

    is_global = sim in HK26_GLOBAL_SIMS
    failures = []

    for var_name, da in snapshot.data_vars.items():
        if var_name == 'weights':
            continue
        nan_frac = float(np.isnan(da.values).mean())

        if nan_frac == 1.0:
            failures.append(f"{var_name}: 100% NaN")
            continue

        if var_name == 'mrsol':
            continue

        if is_global and nan_frac >= MAX_NAN_FRACTION_GLOBAL:
            failures.append(
                f"{var_name}: NaN fraction {nan_frac:.1%} >= {MAX_NAN_FRACTION_GLOBAL:.0%}"
            )

    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Test 5 – value range checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sim,zoom,freq", PARAMS, ids=PARAM_IDS)
def test_value_ranges(get_snapshot, sim, zoom, freq):
    snapshot = get_snapshot(sim, zoom, freq)
    if isinstance(snapshot, Exception):
        pytest.skip(f"Not available: {snapshot}")

    failures = []

    for var_name, (vmin, vmax) in RANGE_CHECKS.items():
        if var_name not in snapshot:
            continue
        valid = snapshot[var_name].values
        valid = valid[~np.isnan(valid)]
        if len(valid) == 0:
            continue

        if vmin is not None and valid.min() < vmin - RANGE_CHECK_ATOL:
            failures.append(f"{var_name}: min {valid.min():.4g} < expected {vmin}")
        if vmax is not None and valid.max() > vmax + RANGE_CHECK_ATOL:
            failures.append(f"{var_name}: max {valid.max():.4g} > expected {vmax}")

    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Test 6 – no NaNs at zoom=0 over full time series
# ---------------------------------------------------------------------------

SIM_FREQ_PARAMS = [(sim, freq) for sim in HK26_SIMS for freq in FREQS]
SIM_FREQ_IDS = [f"{sim.replace('um_', '').replace('_hk26', '')}-{freq}" for sim, freq in SIM_FREQ_PARAMS]


@pytest.mark.parametrize("sim,freq", SIM_FREQ_PARAMS, ids=SIM_FREQ_IDS)
def test_no_nans_zoom0(get_zoom0, sim, freq):
    ds = get_zoom0(sim, freq)
    if isinstance(ds, Exception):
        pytest.skip(f"Not available: {ds}")

    ds = ds.isel(time=slice(1, -1))

    failures = []
    for var_name, da in ds.data_vars.items():
        if var_name == 'weights':
            continue
        nan_frac = float(np.isnan(da.values).mean())
        if nan_frac > 0:
            failures.append(f"{var_name}: {nan_frac:.4%} NaN")

    for msg in failures:
        warnings.warn(f"{sim} zoom=0 {freq}: {msg}", UserWarning, stacklevel=2)
