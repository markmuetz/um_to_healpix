import numpy as np
import pandas as pd
import pytest
import xarray as xr

from um_to_healpix.latlon_to_healpix import LatLon2HealpixRegridder

pytestmark = pytest.mark.slow

REGRID_ZOOM = 2          # must match conftest.py
NCELL = 12 * 4 ** REGRID_ZOOM  # 192


@pytest.fixture(scope='module')
def regridder(tiny_weights):
    return LatLon2HealpixRegridder(
        weights=tiny_weights,
        method='easygems_delaunay',
        zoom_level=REGRID_ZOOM,
        add_cyclic=True,
    )


@pytest.fixture(scope='module')
def regridded_2d(tiny_da, regridder):
    return regridder.regrid(tiny_da, 'longitude', 'latitude')


class TestRegrid2D:
    def test_output_shape(self, regridded_2d):
        assert regridded_2d.dims == ('healpix_index',)
        assert regridded_2d.sizes['healpix_index'] == NCELL

    def test_healpix_index_is_valid_range(self, regridded_2d):
        idx = regridded_2d.healpix_index.values
        assert idx.min() == 0
        assert idx.max() == NCELL - 1

    def test_output_not_all_nan(self, regridded_2d):
        assert not np.isnan(regridded_2d.values).all()

    def test_global_mean_approximately_preserved(self, tiny_da, regridded_2d):
        input_mean = float(tiny_da.mean())
        valid = regridded_2d.values[~np.isnan(regridded_2d.values)]
        output_mean = float(valid.mean())
        assert abs(output_mean - input_mean) / abs(input_mean) < 0.10

    def test_preserves_name(self, tiny_da, regridded_2d):
        assert regridded_2d.name == tiny_da.name


class TestRegrid3D:
    """Verify that a (time, lat, lon) input produces a (time, healpix_index) output."""

    @pytest.fixture(scope='class')
    def da_3d(self, tiny_da):
        ntimes = 3
        time = pd.date_range('2020-01-01', periods=ntimes, freq='h')
        data = np.stack([tiny_da.values] * ntimes, axis=0)
        return xr.DataArray(
            data,
            coords={'time': time, 'latitude': tiny_da.latitude, 'longitude': tiny_da.longitude},
            dims=['time', 'latitude', 'longitude'],
            name='tas',
        )

    def test_output_dims(self, da_3d, regridder):
        result = regridder.regrid(da_3d, 'longitude', 'latitude')
        assert result.dims == ('time', 'healpix_index')

    def test_output_time_size_preserved(self, da_3d, regridder):
        result = regridder.regrid(da_3d, 'longitude', 'latitude')
        assert result.sizes['time'] == da_3d.sizes['time']

    def test_output_healpix_size(self, da_3d, regridder):
        result = regridder.regrid(da_3d, 'longitude', 'latitude')
        assert result.sizes['healpix_index'] == NCELL
