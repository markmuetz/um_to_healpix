"""Fixtures for end-to-end pipeline tests.

The full pipeline is:
  Stage 1 — create_empty_zarr_stores  (pre-allocates stores for all zooms)
  Stage 2 — regrid                    (fills max-zoom store from lat/lon cubes)
  Stage 3 — coarsen_healpix_zarr_region (fills lower-zoom stores)

All zarr stores use the local filesystem; S3 is replaced by a trivial
store_factory (``lambda url: url``).  Orography generation is patched out
with synthetic data so no real .pp or .nc files are needed.
"""
from unittest.mock import patch

import iris
import iris.coords
import iris.cube
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from um_to_healpix.cube_to_da_mapping import MapItem
from um_to_healpix.healpix_coarsen import coarsen_healpix_zarr_region
from um_to_healpix.um_process_tasks import UMProcessTasks, get_crs
from um_to_healpix.util import has_dimensions

MAX_ZOOM = 2
NLON = 24
NLAT = 12       # 15° spacing, ±82.5° — avoids exact pole singularity in Delaunay
NTIMES = 4

_LONS = np.linspace(0.0, 360.0 * (1 - 1 / NLON), NLON)
_LATS = np.linspace(-82.5, 82.5, NLAT)


def _make_tas_cube():
    rng = np.random.default_rng(42)
    data = (rng.random((NTIMES, NLAT, NLON)) * 30 + 273).astype(np.float32)
    time_coord = iris.coords.DimCoord(
        np.arange(NTIMES, dtype=float),
        standard_name='time',
        units='hours since 2020-01-20 00:00:00',
    )
    lat_coord = iris.coords.DimCoord(_LATS, standard_name='latitude', units='degrees')
    lon_coord = iris.coords.DimCoord(_LONS, standard_name='longitude', units='degrees')
    return iris.cube.Cube(
        data, standard_name='air_temperature', units='K',
        dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1), (lon_coord, 2)],
    )


def _mock_orog(max_zoom):
    """Synthetic orog/sftlf datasets keyed by zoom level."""
    result = {}
    for zoom in range(max_zoom + 1):
        ncell = 12 * 4 ** zoom
        cells = np.arange(ncell)
        crs = get_crs(zoom)
        result[zoom] = xr.Dataset({
            'orog': xr.DataArray(
                np.zeros(ncell, dtype=np.float32),
                dims=['healpix_index'], coords={'healpix_index': cells},
            ).assign_coords(crs=crs),
            'sftlf': xr.DataArray(
                np.full(ncell, 50.0, dtype=np.float32),
                dims=['healpix_index'], coords={'healpix_index': cells},
            ).assign_coords(crs=crs),
        })
    return result


def _make_config(tmp_path):
    time2d = pd.date_range('2020-01-20', periods=NTIMES, freq='h')
    return {
        'name': 'e2e_test',
        'regional': False,
        'add_cyclic': True,
        'weightsdir': tmp_path / 'weights',
        'orog_land_sea': None,   # replaced by mock; never read from disk
        'max_zoom': MAX_ZOOM,
        'zarr_store_url_tpl': str(tmp_path / '{freq}_z{zoom}.zarr'),
        'drop_vars': [],
        'groups': {
            '2d': {
                'time': time2d,
                'zarr_store': 'PT1H',
                'name_map': {
                    ('tas', 'air_temperature'): MapItem('air_temperature'),
                },
                'constraint': has_dimensions('time', 'latitude', 'longitude'),
                'chunks': {z: (NTIMES, 12 * 4 ** z) for z in range(MAX_ZOOM + 1)},
            },
        },
        'metadata': {'simulation': 'e2e_test'},
    }


@pytest.fixture(scope='module')
def e2e_pipeline(tmp_path_factory):
    """Run all three pipeline stages once; yield (config, tmp_path).

    Slow: generates Delaunay weights for REGRID_ZOOM=2.
    """
    tmp_path = tmp_path_factory.mktemp('e2e')
    config = _make_config(tmp_path)
    cubes = iris.cube.CubeList([_make_tas_cube()])
    task = {'inpaths': [], 'config_key': 'e2e_test', 'date': '2020-01-20'}

    proc = UMProcessTasks(config, shared_metadata={}, store_factory=lambda url: url)

    # Stage 1: pre-allocate zarr stores at all zooms
    with patch.object(proc, '_gen_orog_land_sea', return_value=_mock_orog(MAX_ZOOM)):
        proc.create_empty_zarr_stores(task, cubes=cubes)

    # Stage 2: regrid lat/lon → healpix at max zoom
    proc.regrid(task, cubes=cubes)

    # Stage 3: coarsen max_zoom → max_zoom-1 → … → 0
    chunks = config['groups']['2d']['chunks']
    url_tpl = config['zarr_store_url_tpl']
    for tgt_zoom in range(MAX_ZOOM - 1, -1, -1):
        src_url = url_tpl.format(freq='PT1H', zoom=tgt_zoom + 1)
        tgt_url = url_tpl.format(freq='PT1H', zoom=tgt_zoom)
        src_ds = xr.open_zarr(src_url)
        coarsen_healpix_zarr_region(src_ds, tgt_url, tgt_zoom, '2d', 0, NTIMES, chunks)

    return config, tmp_path
