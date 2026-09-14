import importlib.util
import sys
from functools import partial
import asyncio
import random
import subprocess as sp
import time
from contextlib import contextmanager
from pathlib import Path

import botocore.exceptions
import iris
import xarray as xr
from easygems import healpix as egh
from iris.experimental.stratify import relevel
from loguru import logger
import numpy as np
import stratify

# import xarray as xr
# async def async_retry_open_zarr(url, max_retries=20):
#     retries = 0
#     while retries < max_retries:
#         try:
#             ds = xr.open_zarr(url)
#             logger.debug(f'Successfully opened {url}')
#             return ds
#         except Exception as e:
#             # This has started (30/4/2025) raising exceptions
#             # It has previously been fine.
#             logger.warning(f'Failed to open {url}')
#             logger.warning(e)
#             retries += 1
#             # Sleep 10s, then 20s... with 5s jitter.
#             timeout = 1 * retries + random.uniform(-0.5, 0.5)
#             logger.warning(f'sleeping for {timeout} s')
#             await asyncio.sleep(timeout)
#     raise Exception(f'failed to open {url} after {retries} retries')


def retry_on_s3_error(fn, *args, what='S3 operation', max_retries=6, base_sleep=30, **kwargs):
    """Call fn(*args, **kwargs), retrying S3/network failures with a growing sleep.

    Reads and store-opens had no retries, so a momentary failure killed the task: the 2026-09-13 outages
    (10 and 16 min) cost 38 coarsen batches this way. The default schedule (30, 60, 120, 240, 480 s + jitter,
    ~15 min total) rides out an outage of that length. Writes have their own retries in
    async_da_to_zarr_with_retries.
    """
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except (botocore.exceptions.ClientError, botocore.exceptions.ConnectionError,
                botocore.exceptions.HTTPClientError, OSError) as e:
            if attempt == max_retries - 1:
                raise
            timeout = max(0.0, base_sleep * 2 ** attempt + random.uniform(-5, 5))
            logger.warning(f'{what} failed ({type(e).__name__}: {str(e)[:120]}); '
                           f'retry {attempt + 1}/{max_retries - 1} in {timeout:.0f}s')
            time.sleep(timeout)


@contextmanager
def task_log(path, level='DEBUG'):
    """Also write loguru output to a human-readable per-task log file (appended to, one file per task)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(path, level=level, mode='a')
    try:
        yield path
    finally:
        logger.remove(sink_id)


async def async_da_to_zarr_with_retries(da, store, region, max_retries=5):
    # This got complicated quite quickly. I was getting exceptions intermittently with da.to_zarr. Handling these
    # exceptions wasn't working because it was being thrown from async code, so I needed to write my own async func
    # so I could call await asyncio.sleep(...). This is how you call it. ChatGPT helped with the async stuff.
    retries = 0
    success = False
    while retries < max_retries:
        try:
            da.to_zarr(store, region=region)
            success = True
            logger.debug(f'{da.name} successfully written to zarr')
            break
        except (botocore.exceptions.ClientError, OSError, PermissionError, FileNotFoundError) as e:
            # This has started (26/4/2025) raising exceptions, one as an inner, one as outer.
            # Not sure which exception is responsible/the one to catch.
            # It has previously been fine.
            logger.warning(f'Failed to write {da.name} to zarr store {store}')
            logger.warning(e)
            retries += 1
            # Sleep 10s, then 20s... with 5s jitter.
            timeout = 10 * retries + random.uniform(-5, 5)
            logger.debug(f'sleeping for {timeout} s')
            await asyncio.sleep(timeout)
    if not success:
        raise Exception(f'failed to write {da.name} to zarr store {store} after {retries} retries')


def model_level_to_pressure(cube, p, z, enforce_greater_than_zero=True, time_indices=None, dtype=np.float32):
    """Interpolate a model-level cube onto pressure levels.

    time_indices selects which time steps to do (default: all). Doing one step at a time keeps the output array
    small: the full (12, 25, 3841, 5120) array is 44 GB in float64 and was the main reason a regrid task peaked at
    ~95 GB. dtype float32 matches the zarr stores the result is written to (float64 precision is discarded there).
    """
    logger.debug(f're-level model level to pressure for {cube.name()}')
    cube = cube[-p.shape[0]:]
    assert (p.coord('time').points == cube.coord('time').points).all()
    if time_indices is None:
        time_indices = range(cube.shape[0])
    time_indices = list(time_indices)

    # Direction of pressure_levels must match that of air_pressure/p.
    # This runs, but it also inverts the 3D fields! Fix by inverting output.
    pressure_levels = z.coord('pressure').points[::-1] * 100  # convert from hPa to Pa.
    interpolator = partial(stratify.interpolate,
                           interpolation=stratify.INTERPOLATE_LINEAR,
                           extrapolation=stratify.EXTRAPOLATE_LINEAR,
                           rising=False)
    new_cube_data = np.zeros((len(time_indices), len(pressure_levels), cube.shape[2], cube.shape[3]), dtype=dtype)
    for out_i, i in enumerate(time_indices):
        logger.trace(i)
        regridded_cube = relevel(cube[i], p[i], pressure_levels, interpolator=interpolator)
        # logger.trace(f'regridded_cube.data.sum() {regridded_cube.data.sum()}')
        # Fix 3D fields so that they are the right way round - invert output.
        new_cube_data[out_i] = regridded_cube.data[::-1]
        del regridded_cube

    if enforce_greater_than_zero:
        # Some values are ending up as negatives (why? Perhaps due to linear extrap. outside domain?)
        # Enfore greater than zero if so (these are all for mass_ fields - must by >= 0).
        new_cube_data[new_cube_data < 0] = 0

    time_coord = cube[time_indices].coord('time') if len(time_indices) < cube.shape[0] else cube.coord('time')
    coords = [(time_coord, 0), (z.coord('pressure'), 1), (z.coord('latitude'), 2),
              (z.coord('longitude'), 3)]
    new_cube = iris.cube.Cube(new_cube_data,
                              long_name=cube.name(),
                              units=cube.units,
                              dim_coords_and_dims=coords,
                              attributes=cube.attributes)
    logger.trace(new_cube)
    return new_cube


def sysrun(cmd):
    return sp.run(cmd, check=True, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf8')


def load_config(file_path):
    path = Path(file_path)
    module_name = path.stem  # Use filename without extension

    # 1. Create the spec
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    # 2. Create a new module based on the spec
    config_module = importlib.util.module_from_spec(spec)

    # 3. Add to sys.modules (optional but recommended for imports)
    sys.modules[module_name] = config_module

    # 4. Execute the module to populate it
    spec.loader.exec_module(config_module)

    return config_module


def exception_info(ex_type, value, tb):
    """Drop user into a debug shell on exception."""
    import traceback

    traceback.print_exception(ex_type, value, tb)
    try:
        # Might not be installed.
        import ipdb as debug
    except ImportError:
        import pdb as debug
    debug.pm()


def has_dimensions(*dims):
    """Returns an Iris constraint that filters cubes based on dimensions."""

    def dim_filter(cube):
        cube_dims = tuple([c.name() for c in cube.dim_coords])
        return cube_dims == dims

    return iris.Constraint(cube_func=dim_filter)


def cube_cell_method_is_not_empty(cube):
    return cube.cell_methods != tuple()


def cube_cell_method_is_empty(cube):
    return cube.cell_methods == tuple()


def invert_cube_sign(cube):
    cube.data = -1 * cube.data
    return cube


def make_percentage(cube):
    cube.data = 100 * cube.data
    return cube


def check_cube_time_length(cube):
    # Shorten cube if it has length 13 (applies to first cube only I think).
    if cube.shape[0] == 13:
        cube = cube[1:]
    return cube


def open_remote_dataset(config, sim, freq, zoom, on_jasmin=False):
    if on_jasmin:
        protocol = 'http'
        baseurl = 'hackathon-o.s3.jc.rl.ac.uk'
    else:
        protocol = 'https'
        baseurl = 'hackathon-o.s3-ext.jc.rl.ac.uk'
    url = f'{protocol}://{baseurl}/sim-data/{config.deploy}/{config.output_vn}/{sim}/um.{freq}.hp_z{zoom}.zarr/'
    print(url)

    ds = xr.open_dataset(url, engine='zarr')
    ds = ds.pipe(attach_coords)
    return ds


def attach_coords(ds: xr.Dataset, signed_lon=False):
    # Same as egh.attach_coords but swap cell for healpix_index.
    ds = egh.fix_crs(ds)

    healpix_index = ds.get("healpix_index") if "healpix_index" in ds.dims else np.arange(egh.get_npix(ds))

    lons, lats = egh.healpix.pix2ang(egh.get_nside(ds), healpix_index, nest=egh.get_nest(ds), lonlat=True)
    if signed_lon:
        lons = np.where(lons <= 180, lons, lons - 360)
    else:
        # Both healpy and healpix produce longitudes in the range [-45, 360]
        # While this is mathematically valid, it may be unexpected in Earth system science.
        lons %= 360
    return ds.assign_coords(
        healpix_index=healpix_index,
        lat=(
            ("healpix_index",),
            lats,
            {"units": "degree_north", "standard_name": "latitude", "axis": "Y"},
        ),
        lon=(
            ("healpix_index",),
            lons,
            {"units": "degree_east", "standard_name": "longitude", "axis": "X"},
        ),
    )
