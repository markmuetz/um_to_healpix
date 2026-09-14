from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytestmark = pytest.mark.e2e

MAX_ZOOM = 2   # must match conftest.py
NTIMES = 4     # must match conftest.py
EXPECTED_VARS = {'tas', 'orog', 'sftlf'}
ZOOMS = list(range(MAX_ZOOM + 1))


class TestZarrStoresCreated:
    """Stage 1: create_empty_zarr_stores produces stores for every zoom."""

    def test_zarr_dirs_exist_for_all_zooms(self, e2e_pipeline):
        config, tmp_path = e2e_pipeline
        url_tpl = config['zarr_store_url_tpl']
        for zoom in ZOOMS:
            path = Path(url_tpl.format(freq='PT1H', zoom=zoom))
            assert path.exists(), f'zarr store missing at zoom={zoom}'

    def test_all_expected_variables_present(self, e2e_pipeline):
        config, _ = e2e_pipeline
        url_tpl = config['zarr_store_url_tpl']
        for zoom in ZOOMS:
            ds = xr.open_zarr(url_tpl.format(freq='PT1H', zoom=zoom))
            assert EXPECTED_VARS.issubset(set(ds.data_vars)), (
                f'zoom={zoom}: expected {EXPECTED_VARS}, got {set(ds.data_vars)}'
            )

    def test_cell_counts_match_healpix_formula(self, e2e_pipeline):
        config, _ = e2e_pipeline
        url_tpl = config['zarr_store_url_tpl']
        for zoom in ZOOMS:
            ds = xr.open_zarr(url_tpl.format(freq='PT1H', zoom=zoom))
            expected_ncell = 12 * 4 ** zoom
            assert ds.sizes['healpix_index'] == expected_ncell, (
                f'zoom={zoom}: expected {expected_ncell} cells, got {ds.sizes["healpix_index"]}'
            )

    def test_time_size_matches_config(self, e2e_pipeline):
        config, _ = e2e_pipeline
        url_tpl = config['zarr_store_url_tpl']
        ds = xr.open_zarr(url_tpl.format(freq='PT1H', zoom=MAX_ZOOM))
        assert ds.sizes['time'] == NTIMES


class TestRegrid:
    """Stage 2: regrid populates the max-zoom zarr store."""

    def test_tas_not_all_nan_at_max_zoom(self, e2e_pipeline):
        config, _ = e2e_pipeline
        url = config['zarr_store_url_tpl'].format(freq='PT1H', zoom=MAX_ZOOM)
        ds = xr.open_zarr(url).compute()
        assert not np.isnan(ds['tas'].values).all()

    def test_tas_values_in_physically_plausible_range(self, e2e_pipeline):
        config, _ = e2e_pipeline
        url = config['zarr_store_url_tpl'].format(freq='PT1H', zoom=MAX_ZOOM)
        ds = xr.open_zarr(url).compute()
        valid = ds['tas'].values[~np.isnan(ds['tas'].values)]
        assert len(valid) > 0
        assert valid.min() > 200      # > −73 °C
        assert valid.max() < 350      # < 77 °C


class TestCoarsen:
    """Stage 3: coarsening fills lower-zoom stores with correct structure."""

    @pytest.mark.parametrize('zoom', list(range(MAX_ZOOM)))
    def test_coarsened_tas_not_all_nan(self, e2e_pipeline, zoom):
        config, _ = e2e_pipeline
        url = config['zarr_store_url_tpl'].format(freq='PT1H', zoom=zoom)
        ds = xr.open_zarr(url).compute()
        assert not np.isnan(ds['tas'].values).all(), f'all NaN at zoom={zoom}'

    @pytest.mark.parametrize('zoom', list(range(MAX_ZOOM)))
    def test_coarsened_cell_count(self, e2e_pipeline, zoom):
        config, _ = e2e_pipeline
        url = config['zarr_store_url_tpl'].format(freq='PT1H', zoom=zoom)
        ds = xr.open_zarr(url)
        assert ds.sizes['healpix_index'] == 12 * 4 ** zoom

    def test_coarsened_mean_close_to_max_zoom_mean(self, e2e_pipeline):
        config, _ = e2e_pipeline
        url_tpl = config['zarr_store_url_tpl']
        ds_max = xr.open_zarr(url_tpl.format(freq='PT1H', zoom=MAX_ZOOM)).compute()
        ds_z0 = xr.open_zarr(url_tpl.format(freq='PT1H', zoom=0)).compute()
        max_mean = float(ds_max['tas'].mean())
        z0_mean = float(ds_z0['tas'].mean())
        assert abs(z0_mean - max_mean) / abs(max_mean) < 0.05
