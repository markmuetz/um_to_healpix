import operator

import iris
import iris.coords
import iris.cube
import numpy as np
import pytest

from um_to_healpix.cube_to_da_mapping import DataArrayExtractor, MapItem, MultiMapItem


def _make_cube(ntimes=12, name='air_temperature'):
    """Minimal iris cube with time/lat/lon dims."""
    data = np.zeros((ntimes, 3, 4), dtype=np.float32)
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


class TestMapItem:
    def test_repr_contains_class_name(self):
        item = MapItem('air_temperature')
        assert 'MapItem' in repr(item)

    def test_extra_attrs_defaults_to_empty_dict(self):
        item = MapItem('air_temperature')
        assert item.extra_attrs == {}

    def test_extra_processing_defaults_to_none(self):
        item = MapItem('air_temperature')
        assert item.extra_processing is None


class TestMultiMapItem:
    def test_repr_contains_class_name(self):
        item = MultiMapItem([MapItem('a'), MapItem('b')], ops=[operator.add])
        assert 'MultiMapItem' in repr(item)

    def test_extra_attrs_defaults_to_empty_dict(self):
        item = MultiMapItem([MapItem('a')], ops=[])
        assert item.extra_attrs == {}


class TestDataArrayExtractor:
    def test_extract_da_trims_13_timesteps_to_12(self):
        cube = _make_cube(ntimes=13)
        cubes = iris.cube.CubeList([cube])
        map_item = MapItem('air_temperature')
        extractor = DataArrayExtractor(None, None)
        da = extractor.extract_da(map_item, cubes)
        assert da.shape[0] == 12

    def test_extract_da_leaves_12_timesteps_unchanged(self):
        cube = _make_cube(ntimes=12)
        cubes = iris.cube.CubeList([cube])
        map_item = MapItem('air_temperature')
        extractor = DataArrayExtractor(None, None)
        da = extractor.extract_da(map_item, cubes)
        assert da.shape[0] == 12

    def test_extract_cubes_raises_on_missing_constraint(self):
        cube = _make_cube(name='air_temperature')
        cubes = iris.cube.CubeList([cube])
        map_item = MapItem('surface_temperature')  # not in cubes
        extractor = DataArrayExtractor(None, None)
        with pytest.raises(iris.exceptions.ConstraintMismatchError):
            extractor.extract_cubes(map_item, cubes)
