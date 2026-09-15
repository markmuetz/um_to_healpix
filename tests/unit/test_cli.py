"""Tests for um_to_healpix.cli helpers and the pp scan they rely on."""
from pathlib import Path

import iris
import iris.coords
import iris.cube
import numpy as np
import pandas as pd
import pytest

from um_to_healpix import cli
from um_to_healpix.pp_scan import DEFAULT_PP_GLOB, find_dyamond3_pp_dates_to_paths, parse_date_from_pp_path


def _cube(name='x_wind', stash='m01s03i225', level=None, nlon=8, nlat=6, lon0=0.035, lat0=-89.977):
    lon = iris.coords.DimCoord(np.linspace(lon0, 359.965, nlon), standard_name='longitude', units='degrees')
    lat = iris.coords.DimCoord(np.linspace(lat0, -lat0, nlat), standard_name='latitude', units='degrees')
    cube = iris.cube.Cube(np.zeros((nlat, nlon), dtype=np.float32), long_name=name, units='m s-1',
                          dim_coords_and_dims=[(lat, 0), (lon, 1)])
    cube.attributes['STASH'] = stash
    if level:
        # load_raw gives one cube per (time, level), so the level is a *scalar* coord.
        cube.add_aux_coord(iris.coords.AuxCoord([500.0], long_name=level))
    return cube


class TestLevels:
    def test_scalar_level_coord_is_found(self):
        """load_raw puts the level on a scalar coord; looking only at dim coords misses it."""
        assert cli._levels_of(_cube(level='pressure')) == 'pressure'
        assert cli._levels_of(_cube()) == '-'

    def test_at_level(self):
        assert cli._at_level(_cube(level='pressure'), 'pressure')
        assert not cli._at_level(_cube(level='pressure'), 'model_level_number')
        assert cli._at_level(_cube(), None)          # surface field, no level expected
        assert not cli._at_level(_cube(level='pressure'), None)


class TestExtractFirst:
    """ua/uas, va/vas, ta/tas and hus/huss share a cube name, so only the level tells them apart."""

    def _cubes(self):
        return iris.cube.CubeList([_cube(), _cube(level='pressure', stash='m01s15i201')])

    def test_level_disambiguates_2d_from_3d(self):
        from um_to_healpix.cube_to_da_mapping import MapItem
        item = MapItem('x_wind')
        surface, note = cli._extract_first(item, self._cubes(), None)
        assert note is None and surface[0].attributes['STASH'] == 'm01s03i225'
        pressure, note = cli._extract_first(item, self._cubes(), 'pressure')
        assert note is None and pressure[0].attributes['STASH'] == 'm01s15i201'

    def test_no_cube_at_that_level_is_reported(self):
        from um_to_healpix.cube_to_da_mapping import MapItem
        found, note = cli._extract_first(MapItem('x_wind'), self._cubes(), 'depth')
        assert found == [] and 'none at depth' in note

    def test_missing_constraint_is_reported(self):
        from um_to_healpix.cube_to_da_mapping import MapItem
        found, note = cli._extract_first(MapItem('nonexistent_name'), self._cubes(), None)
        assert found == [] and note == 'no cube matches the constraint'


class TestGridOf:
    def test_matches_the_pipeline_weights_filename(self):
        """The reported filename must be exactly what the regrid step looks for."""
        from um_to_healpix.um_process_tasks import weights_filename
        import xarray as xr
        cube = _cube()
        grid, fn = cli._grid_of(cube, 10)
        assert '8x6' in grid and 'lon 0.035' in grid
        da = xr.DataArray(np.zeros((6, 8)),
                          coords={'latitude': cube.coord('latitude').points,
                                  'longitude': cube.coord('longitude').points},
                          dims=['latitude', 'longitude'])
        assert fn == weights_filename(da, 10, 'longitude', 'latitude', True, False)

    def test_no_horizontal_coords(self):
        cube = iris.cube.Cube(np.zeros(3), long_name='scalar')
        assert cli._grid_of(cube, 10) == (None, None)


class TestPpScan:
    @pytest.mark.parametrize('name,expected', [
        ('glm.n2560_CoMA9_p4k.apvera_20200120T0000Z.pp', '2020-01-20 00:00'),
        ('glm.n2560_RAL3p3.apvera_20200120T00.pp', '2020-01-20 00:00'),
    ])
    def test_parse_date(self, name, expected):
        assert parse_date_from_pp_path(Path(name)) == pd.Timestamp(expected)

    def test_new_layout_needs_its_glob(self, tmp_path):
        """The +4K reruns use glm/apver*/*.pp; the default field.pp glob finds nothing."""
        for stream in ('apvera', 'apverb', 'apverc', 'apverd'):
            d = tmp_path / stream
            d.mkdir()
            (d / f'glm.sim.{stream}_20200120T0000Z.pp').touch()
        assert find_dyamond3_pp_dates_to_paths(tmp_path, DEFAULT_PP_GLOB) == {}
        found = find_dyamond3_pp_dates_to_paths(tmp_path, 'apve*/*.pp')
        assert list(found) == [pd.Timestamp('2020-01-20')] and len(found[pd.Timestamp('2020-01-20')]) == 4

    def test_incomplete_dates_dropped(self, tmp_path):
        for stream in ('apvera', 'apverb'):
            d = tmp_path / stream
            d.mkdir()
            (d / f'glm.sim.{stream}_20200120T0000Z.pp').touch()
        assert find_dyamond3_pp_dates_to_paths(tmp_path, 'apve*/*.pp') == {}

    def test_apvere_ignored(self, tmp_path):
        for stream in ('apvera', 'apverb', 'apverc', 'apverd', 'apvere'):
            d = tmp_path / stream
            d.mkdir()
            (d / f'glm.sim.{stream}_20200120T0000Z.pp').touch()
        found = find_dyamond3_pp_dates_to_paths(tmp_path, 'apve*/*.pp')
        assert len(found[pd.Timestamp('2020-01-20')]) == 4
        # p.name, not str(p): the tmp_path directory is itself called test_apvere_ignored0.
        assert not any('apvere' in p.name for p in found[pd.Timestamp('2020-01-20')])
