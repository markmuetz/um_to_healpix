import numpy as np
import pytest
import xarray as xr

from um_to_healpix.latlon_to_healpix import (
    LatLon2HealpixRegridder,
    _xr_add_cyclic_point,
    get_extent,
    get_regional_cell_idx,
)


def _make_latlon_da(nlat=5, nlon=8, name='tas'):
    lons = np.linspace(0.0, 360.0 * (1 - 1 / nlon), nlon)
    lats = np.linspace(-90.0, 90.0, nlat)
    data = np.ones((nlat, nlon), dtype=np.float32)
    return xr.DataArray(
        data,
        coords={'latitude': lats, 'longitude': lons},
        dims=['latitude', 'longitude'],
        name=name,
    )


def _minimal_weights():
    """Minimal weights Dataset sufficient to construct an easygems_delaunay regridder."""
    return xr.Dataset({
        'weights': xr.DataArray(np.array([1.0], dtype=np.float32)),
        'src_idx': xr.DataArray(np.array([0], dtype=np.int64)),
        'tgt_idx': xr.DataArray(np.array([0], dtype=np.int64)),
    })


class TestAddCyclicPoint:
    def test_adds_one_longitude_column(self):
        da = _make_latlon_da(nlon=8)
        result = _xr_add_cyclic_point(da, 'longitude')
        assert result.sizes['longitude'] == da.sizes['longitude'] + 1

    def test_wrap_column_values_match_first_column(self):
        da = _make_latlon_da(nlon=8)
        # Give each longitude column a unique value so we can verify the wrap.
        da.values[:] = np.arange(da.sizes['longitude'])[np.newaxis, :]
        result = _xr_add_cyclic_point(da, 'longitude')
        np.testing.assert_array_equal(result.values[:, -1], da.values[:, 0])

    def test_preserves_data_array_name(self):
        da = _make_latlon_da(nlon=8, name='pr')
        result = _xr_add_cyclic_point(da, 'longitude')
        assert result.name == 'pr'

    def test_latitude_dimension_unchanged(self):
        da = _make_latlon_da(nlat=6, nlon=8)
        result = _xr_add_cyclic_point(da, 'longitude')
        assert result.sizes['latitude'] == 6


class TestGetExtent:
    def test_returns_correct_bounds(self):
        lons = np.array([10.0, 20.0, 30.0])
        lats = np.array([-5.0, 0.0, 5.0])
        da = xr.DataArray(
            np.zeros((3, 3)),
            coords={'latitude': lats, 'longitude': lons},
            dims=['latitude', 'longitude'],
        )
        minlon, maxlon, minlat, maxlat = get_extent(da, 'longitude', 'latitude')
        assert minlon == pytest.approx(10.0)
        assert maxlon == pytest.approx(30.0)
        assert minlat == pytest.approx(-5.0)
        assert maxlat == pytest.approx(5.0)


class TestGetRegionalCellIdx:
    def test_returns_non_empty_for_valid_box(self):
        extent = [0.0, 90.0, -45.0, 45.0]
        _, _, icell = get_regional_cell_idx(extent, zoom=1)
        assert len(icell) > 0

    def test_returned_cells_lie_within_extent(self):
        extent = [0.0, 90.0, -45.0, 45.0]
        hp_lon, hp_lat, icell = get_regional_cell_idx(extent, zoom=1)
        assert (hp_lon[icell] > extent[0]).all()
        assert (hp_lon[icell] < extent[1]).all()
        assert (hp_lat[icell] > extent[2]).all()
        assert (hp_lat[icell] < extent[3]).all()

    def test_wrapped_longitude_africa_domain(self):
        # extent[1] > 360 triggers the hp_lon wrap branch
        extent = [300.0, 415.0, -40.0, 40.0]
        _, _, icell = get_regional_cell_idx(extent, zoom=1)
        assert len(icell) > 0


class TestLatLon2HealpixRegridder:
    def test_raises_on_invalid_method(self):
        with pytest.raises(ValueError, match='method must be'):
            LatLon2HealpixRegridder(weights=None, method='invalid_method')

    def test_accepts_easygems_delaunay(self):
        r = LatLon2HealpixRegridder(weights=_minimal_weights(), method='easygems_delaunay')
        assert r.method == 'easygems_delaunay'

    def test_accepts_easygems_delaunay_parallel(self):
        r = LatLon2HealpixRegridder(weights=_minimal_weights(), method='easygems_delaunay_parallel')
        assert r.method == 'easygems_delaunay_parallel'

    def test_accepts_earth2grid(self):
        # earth2grid does not use weights at init time
        r = LatLon2HealpixRegridder(weights=None, method='earth2grid')
        assert r.method == 'earth2grid'
