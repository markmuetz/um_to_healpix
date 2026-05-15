import numpy as np
import pandas as pd
import pytest
import xarray as xr

from um_to_healpix.healpix_coarsen import (
    map_global_to_regional,
    map_regional_to_global,
    nan_weight,
)


class TestNanWeight:
    def test_all_valid_returns_one(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert nan_weight(arr, axis=0) == pytest.approx(1.0)

    def test_half_nan_returns_half(self):
        arr = np.array([1.0, np.nan, 3.0, np.nan])
        assert nan_weight(arr, axis=0) == pytest.approx(0.5)

    def test_all_nan_returns_zero(self):
        arr = np.full(4, np.nan)
        assert nan_weight(arr, axis=0) == pytest.approx(0.0)

    def test_2d_array_reduces_along_axis(self):
        arr = np.array([
            [1.0, np.nan, 3.0, np.nan],  # 2/4 valid → 0.5
            [1.0, 2.0,   3.0, 4.0  ],   # 4/4 valid → 1.0
        ])
        result = nan_weight(arr, axis=1)
        np.testing.assert_array_almost_equal(result, [0.5, 1.0])

    def test_output_is_float32(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert nan_weight(arr, axis=0).dtype == np.float32


class TestMapRegionalToGlobal:
    def _make_regional_da(self, regional_cells, ntimes=3, name='tas'):
        time = pd.date_range('2020-01-01', periods=ntimes, freq='h')
        data = np.ones((ntimes, len(regional_cells)), dtype=np.float32)
        return xr.DataArray(
            data, dims=['time', 'healpix_index'],
            coords={'time': time, 'healpix_index': regional_cells},
            name=name,
        )

    def test_output_has_global_cell_count(self):
        src_zoom = 2
        ncell_global = 12 * 4 ** src_zoom
        da = self._make_regional_da(regional_cells=np.array([0, 1, 2, 3, 4]))
        result = map_regional_to_global(da, src_zoom=src_zoom, dim='2d')
        assert result.sizes['healpix_index'] == ncell_global

    def test_regional_cells_have_data_rest_are_nan(self):
        src_zoom = 2
        regional_cells = np.array([0, 1, 2, 3, 4])
        da = self._make_regional_da(regional_cells=regional_cells)
        result = map_regional_to_global(da, src_zoom=src_zoom, dim='2d')
        np.testing.assert_array_equal(
            result.isel(healpix_index=slice(0, 5)).values,
            np.ones((3, 5), dtype=np.float32),
        )
        assert np.isnan(result.isel(healpix_index=slice(5, None)).values).all()

    def test_weights_variable_produces_1d_output(self):
        src_zoom = 2
        ncell_global = 12 * 4 ** src_zoom
        regional_cells = np.array([10, 11, 12])
        da = xr.DataArray(
            np.array([0.8, 0.9, 1.0], dtype=np.float32),
            dims=['healpix_index'],
            coords={'healpix_index': regional_cells},
            name='weights',
        )
        result = map_regional_to_global(da, src_zoom=src_zoom, dim='2d')
        assert result.dims == ('healpix_index',)
        assert result.sizes['healpix_index'] == ncell_global
        np.testing.assert_array_equal(
            result.isel(healpix_index=slice(10, 13)).values,
            np.array([0.8, 0.9, 1.0], dtype=np.float32),
        )


class TestMapGlobalToRegional:
    def test_output_has_regional_cell_count(self):
        tgt_zoom = 2
        ncell_global = 12 * 4 ** tgt_zoom
        regional_cells = np.array([5, 6, 7, 8])
        time = pd.date_range('2020-01-01', periods=2, freq='h')

        data = np.tile(np.arange(ncell_global, dtype=np.float32), (2, 1))
        da = xr.DataArray(
            data, dims=['time', 'healpix_index'],
            coords={'time': time, 'healpix_index': np.arange(ncell_global)},
            name='tas',
        )
        src_ds_time_slice = xr.Dataset({'tas': da})
        tgt_ds_store = xr.Dataset(coords={'healpix_index': regional_cells})

        result = map_global_to_regional(da, src_ds_time_slice, tgt_ds_store, dim='2d')
        assert result.sizes['healpix_index'] == len(regional_cells)

    def test_output_values_match_global_at_target_indices(self):
        tgt_zoom = 2
        ncell_global = 12 * 4 ** tgt_zoom
        regional_cells = np.array([5, 6, 7, 8])
        time = pd.date_range('2020-01-01', periods=2, freq='h')

        # Values equal cell index so we can verify selection easily
        data = np.tile(np.arange(ncell_global, dtype=np.float32), (2, 1))
        da = xr.DataArray(
            data, dims=['time', 'healpix_index'],
            coords={'time': time, 'healpix_index': np.arange(ncell_global)},
            name='tas',
        )
        src_ds_time_slice = xr.Dataset({'tas': da})
        tgt_ds_store = xr.Dataset(coords={'healpix_index': regional_cells})

        result = map_global_to_regional(da, src_ds_time_slice, tgt_ds_store, dim='2d')
        expected = data[:, regional_cells]
        np.testing.assert_array_equal(result.values, expected)
