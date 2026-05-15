import iris
import iris.coords
import iris.cube
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from um_to_healpix.util import (
    has_dimensions,
    invert_cube_sign,
    make_percentage,
)
from um_to_healpix.um_process_tasks import find_halfpast_time, weights_filename


def _make_latlon_da(nlat=3, nlon=4):
    lons = np.array([0.0, 90.0, 180.0, 270.0])
    lats = np.array([-30.0, 0.0, 30.0])
    return xr.DataArray(
        np.zeros((nlat, nlon)),
        coords={'latitude': lats, 'longitude': lons},
        dims=['latitude', 'longitude'],
    )


def _make_cube(ntimes=3, name='air_temperature'):
    data = np.ones((ntimes, 3, 4), dtype=np.float32)
    time_coord = iris.coords.DimCoord(
        np.arange(ntimes, dtype=float),
        standard_name='time',
        units='hours since 2020-01-01 00:00:00',
    )
    lat_coord = iris.coords.DimCoord(
        np.array([-30.0, 0.0, 30.0]), standard_name='latitude', units='degrees',
    )
    lon_coord = iris.coords.DimCoord(
        np.array([0.0, 90.0, 180.0, 270.0]), standard_name='longitude', units='degrees',
    )
    return iris.cube.Cube(
        data, standard_name=name, units='K',
        dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1), (lon_coord, 2)],
    )


class TestFindHalfpastTime:
    def test_detects_half_past_times(self):
        # :30 minute times → should return the coord name
        times = pd.date_range('2020-01-01 00:30', periods=3, freq='h')
        da = xr.DataArray(np.zeros(3), coords={'time': times}, dims=['time'])
        assert find_halfpast_time(da) == 'time'

    def test_returns_none_for_on_the_hour(self):
        times = pd.date_range('2020-01-01 00:00', periods=3, freq='h')
        da = xr.DataArray(np.zeros(3), coords={'time': times}, dims=['time'])
        assert find_halfpast_time(da) is None

    def test_ignores_coords_not_starting_with_time(self):
        times = pd.date_range('2020-01-01 00:30', periods=3, freq='h')
        da = xr.DataArray(
            np.zeros((3, 4)),
            coords={'time': times, 'longitude': np.array([0.0, 90.0, 180.0, 270.0])},
            dims=['time', 'longitude'],
        )
        assert find_halfpast_time(da) == 'time'


class TestWeightsFilename:
    def test_contains_zoom_level(self):
        da = _make_latlon_da()
        name = weights_filename(da, zoom=5, lonname='longitude', latname='latitude',
                                add_cyclic=True, regional=False)
        assert 'hpz5' in name

    def test_contains_cyclic_and_regional_flags(self):
        da = _make_latlon_da()
        name = weights_filename(da, zoom=3, lonname='longitude', latname='latitude',
                                add_cyclic=True, regional=False)
        assert 'cyclic_lon=True' in name
        assert 'regional=False' in name

    def test_ends_with_nc(self):
        da = _make_latlon_da()
        name = weights_filename(da, zoom=3, lonname='longitude', latname='latitude',
                                add_cyclic=False, regional=True)
        assert name.endswith('.nc')


class TestInvertCubeSign:
    def test_negates_data(self):
        cube = _make_cube()
        original = cube.data.copy()
        result = invert_cube_sign(cube)
        np.testing.assert_array_equal(result.data, -original)

    def test_returns_same_cube(self):
        cube = _make_cube()
        result = invert_cube_sign(cube)
        assert result is cube


class TestMakePercentage:
    def test_multiplies_by_100(self):
        cube = _make_cube()
        original = cube.data.copy()
        result = make_percentage(cube)
        np.testing.assert_array_equal(result.data, 100 * original)

    def test_returns_same_cube(self):
        cube = _make_cube()
        result = make_percentage(cube)
        assert result is cube


class TestHasDimensions:
    def test_matches_cube_with_correct_dims(self):
        cube = _make_cube()
        constraint = has_dimensions('time', 'latitude', 'longitude')
        result = iris.cube.CubeList([cube]).extract(constraint)
        assert len(result) == 1

    def test_rejects_cube_with_wrong_dims(self):
        cube = _make_cube()
        constraint = has_dimensions('time', 'pressure', 'latitude', 'longitude')
        result = iris.cube.CubeList([cube]).extract(constraint)
        assert len(result) == 0
