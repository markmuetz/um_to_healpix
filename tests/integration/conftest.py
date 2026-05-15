import numpy as np
import pytest
import xarray as xr

from um_to_healpix.latlon_to_healpix import gen_weights

# Zoom level used for weight generation and regrid integration tests.
REGRID_ZOOM = 2

# Grid dimensions — large enough that the convex hull covers all healpix cells
# at REGRID_ZOOM=2 (ncell=192) without NaNs.
# easygems 0.0.14 is required. Versions 0.1.x introduced a stereographic transform
# in compute_weights_delaunay that produces NaN for exact ±90° latitudes, causing
# Delaunay to crash. Pin is recorded in pyproject.toml.

NLON = 24
NLAT = 13   # odd number gives a centre row at lat=0; linspace(-90,90,13) step=15°


@pytest.fixture(scope='session')
def tiny_da():
    """24×13 global lat/lon 2-D DataArray; base for weight generation."""
    lons = np.linspace(0.0, 360.0 * (1 - 1 / NLON), NLON)
    lats = np.linspace(-90.0, 90.0, NLAT)
    # sin(lat)+2 keeps values strictly positive (easier mean checks).
    data = (np.sin(np.deg2rad(lats)) + 2)[:, np.newaxis] * np.ones((NLAT, NLON), dtype=np.float32)
    return xr.DataArray(
        data,
        coords={'latitude': lats, 'longitude': lons},
        dims=['latitude', 'longitude'],
        name='tas',
    )


@pytest.fixture(scope='session')
def tiny_weights(tiny_da, tmp_path_factory):
    """Delaunay weights for tiny_da at REGRID_ZOOM.  Session-scoped (slow)."""
    tmp = tmp_path_factory.mktemp('weights')
    path = tmp / f'weights_z{REGRID_ZOOM}.nc'
    gen_weights(
        tiny_da,
        weights_path=path,
        zoom=REGRID_ZOOM,
        lonname='longitude',
        latname='latitude',
        add_cyclic=True,
    )
    return xr.load_dataset(str(path))
