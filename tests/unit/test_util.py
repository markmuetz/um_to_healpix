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
    model_level_to_pressure,
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


def _make_model_level_cubes(ntime=2, nlev=8, nlat=5, nlon=7, npress=4):
    """A model-level cube, its air_pressure cube, and a target cube carrying the pressure coord."""
    rng = np.random.default_rng(0)
    time_coord = iris.coords.DimCoord(np.arange(ntime, dtype=float), standard_name='time',
                                      units='hours since 2020-01-20 00:00:00')
    lev = iris.coords.DimCoord(np.arange(nlev, dtype=float), long_name='model_level_number')
    lat = iris.coords.DimCoord(np.linspace(-60, 60, nlat), standard_name='latitude', units='degrees')
    lon = iris.coords.DimCoord(np.linspace(0, 300, nlon), standard_name='longitude', units='degrees')
    dims = [(time_coord, 0), (lev, 1), (lat, 2), (lon, 3)]

    # Pressure falls with model level, and varies horizontally, as in the UM.
    profile = np.linspace(100_000, 5_000, nlev)[None, :, None, None]
    noise = 1 + 0.02 * rng.standard_normal((ntime, 1, nlat, nlon))
    p = iris.cube.Cube(profile * noise, long_name='air_pressure', units='Pa', dim_coords_and_dims=dims)
    cube = iris.cube.Cube(rng.random((ntime, nlev, nlat, nlon)) + 0.5, long_name='mass_fraction',
                          units='kg kg-1', dim_coords_and_dims=dims)

    press = iris.coords.DimCoord(np.linspace(1000, 100, npress), long_name='pressure', units='hPa')
    z = iris.cube.Cube(np.zeros((ntime, npress, nlat, nlon)), long_name='geopotential_height', units='m',
                       dim_coords_and_dims=[(time_coord, 0), (press, 1), (lat, 2), (lon, 3)])
    return cube, p, z


class TestModelLevelToPressure:
    """nproc only splits the work; it must not change the answer (see run doc item 8)."""

    def test_parallel_matches_serial_exactly(self):
        cube, p, z = _make_model_level_cubes()
        serial = model_level_to_pressure(cube, p, z, nproc=1)
        parallel = model_level_to_pressure(cube, p, z, nproc=3)
        np.testing.assert_array_equal(serial.data, parallel.data)
        assert serial.shape == parallel.shape

    def test_nproc_not_dividing_columns_still_exact(self):
        """7 columns across 4 workers: uneven splits must still tile the array exactly."""
        cube, p, z = _make_model_level_cubes(nlon=7)
        np.testing.assert_array_equal(model_level_to_pressure(cube, p, z, nproc=1).data,
                                      model_level_to_pressure(cube, p, z, nproc=4).data)

    def test_more_workers_than_columns(self):
        cube, p, z = _make_model_level_cubes(nlon=2)
        np.testing.assert_array_equal(model_level_to_pressure(cube, p, z, nproc=1).data,
                                      model_level_to_pressure(cube, p, z, nproc=6).data)

    def test_time_indices_subset(self):
        cube, p, z = _make_model_level_cubes(ntime=3)
        one = model_level_to_pressure(cube, p, z, time_indices=[1], nproc=2)
        assert one.shape[0] == 1
        np.testing.assert_array_equal(one.data[0], model_level_to_pressure(cube, p, z, nproc=1).data[1])

    def test_enforce_greater_than_zero(self):
        cube, p, z = _make_model_level_cubes()
        cube.data[:] -= 5.0  # force negatives through the extrapolation
        out = model_level_to_pressure(cube, p, z, nproc=2)
        assert (out.data >= 0).all()
