"""Shared utility functions used across the UM-to-HEALPix workflow.

This module collects small, reusable helpers for:
- resilient writes to remote/local Zarr stores,
- model-level to pressure-level interpolation,
- shell command execution,
- dynamic import of Python config files,
- debugging convenience hooks,
- Iris cube filtering and lightweight cube transformations, and
- opening remote simulation datasets from object storage.

Most functions are intentionally thin wrappers around common library calls.
Where behavior is non-obvious (for example retry backoff and level inversion),
extra comments explain the implementation choices and historical context.
"""

import importlib.util
import sys
from functools import partial
import asyncio
import random
import subprocess as sp
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


async def async_da_to_zarr_with_retries(da, store, region, max_retries=5):
    """Write a DataArray to Zarr with retry/backoff for intermittent I/O failures.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to persist.
    store : str | MutableMapping | fsspec-compatible store
        Zarr store target passed directly to ``da.to_zarr``.
    region : dict
        Region selection for partial writes (forwarded to ``to_zarr``).
    max_retries : int, default=5
        Maximum number of write attempts before raising.

    Notes
    -----
    - This function is ``async`` so that retry delays can use
      ``await asyncio.sleep(...)`` without blocking the event loop.
    - The backoff is linear (10 s, 20 s, ...) with random jitter to reduce
      synchronized retries when multiple workers fail at once.
    """
    # Track number of failed attempts so far.
    retries = 0
    # Explicit success flag lets us differentiate clean exit vs exhaustion.
    success = False

    # Keep trying until we either succeed or exhaust the retry budget.
    while retries < max_retries:
        try:
            # Use xarray's native Zarr writer. Any exception below triggers retry.
            da.to_zarr(store, region=region)
            success = True
            logger.debug(f'{da.name} successfully written to zarr')
            break
        except (botocore.exceptions.ClientError, OSError, PermissionError, FileNotFoundError) as e:
            # Catch a broad family of practical storage failures seen in prod:
            # network/client object-store errors and filesystem permission/path issues.
            logger.warning(f'Failed to write {da.name} to zarr store {store}')
            logger.warning(e)
            retries += 1
            # Linear backoff with jitter (seconds): 10, 20, 30, ... ± 5.
            timeout = 10 * retries + random.uniform(-5, 5)
            logger.debug(f'sleeping for {timeout} s')
            await asyncio.sleep(timeout)

    # If we never set success=True, surface a hard failure to the caller.
    if not success:
        raise Exception(f'failed to write {da.name} to zarr store {store} after {retries} retries')


def model_level_to_pressure(cube, p, z, enforce_greater_than_zero=True):
    """Interpolate model-level data onto pressure levels.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input data cube with dimensions expected to include
        ``(time, model_level, latitude, longitude)``.
    p : iris.cube.Cube
        Air pressure cube used as the vertical coordinate source per time slice.
        Its time axis must align with ``cube`` after trimming.
    z : iris.cube.Cube
        Cube containing the target pressure coordinate (in hPa) used to define
        destination levels.
    enforce_greater_than_zero : bool, default=True
        If True, clip negative interpolated values to zero.

    Returns
    -------
    iris.cube.Cube
        New cube on pressure levels with dimensions
        ``(time, pressure, latitude, longitude)``.

    Notes
    -----
    - ``stratify`` expects destination level ordering to be consistent with the
      pressure field ordering; level reversal and output inversion here are a
      practical correction to maintain expected field orientation.
    """
    logger.debug(f're-level model level to pressure for {cube.name()}')

    # Align source cube time length with the pressure cube if needed by slicing
    # the trailing time steps to match p.
    cube = cube[-p.shape[0]:]
    # Guard against misaligned timesteps before interpolation.
    assert (p.coord('time').points == cube.coord('time').points).all()

    # Direction of pressure_levels must match that of air_pressure/p.
    # This runs, but it also inverts the 3D fields! Fix by inverting output.
    # Convert hPa to Pa for consistency with pressure field units.
    pressure_levels = z.coord('pressure').points[::-1] * 100  # convert from hPa to Pa.

    # Build an interpolation function configured for linear interpolation and
    # linear extrapolation. rising=False indicates expected coordinate ordering.
    interpolator = partial(stratify.interpolate,
                           interpolation=stratify.INTERPOLATE_LINEAR,
                           extrapolation=stratify.EXTRAPOLATE_LINEAR,
                           rising=False)

    # Preallocate output for speed and predictable shape:
    # (time, pressure, latitude, longitude).
    new_cube_data = np.zeros((cube.shape[0], len(pressure_levels), cube.shape[2], cube.shape[3]))

    # Interpolate each time slice independently to avoid mixing temporal axes.
    for i in range(cube.shape[0]):
        logger.trace(i)
        regridded_cube = relevel(cube[i], p[i], pressure_levels, interpolator=interpolator)
        # logger.trace(f'regridded_cube.data.sum() {regridded_cube.data.sum()}')
        # Fix 3D fields so that they are the right way round - invert output.
        new_cube_data[i] = regridded_cube.data[::-1]

    if enforce_greater_than_zero:
        # Some values are ending up as negatives (why? Perhaps due to linear extrap. outside domain?)
        # Enfore greater than zero if so (these are all for mass_ fields - must by >= 0).
        # In-place clipping avoids extra array allocations.
        new_cube_data[new_cube_data < 0] = 0

    # Rebuild an Iris cube with the new vertical coordinate while preserving
    # name, units, and source attributes.
    coords = [(cube.coord('time'), 0), (z.coord('pressure'), 1), (cube.coord('latitude'), 2),
              (cube.coord('longitude'), 3)]
    new_cube = iris.cube.Cube(new_cube_data,
                              long_name=cube.name(),
                              units=cube.units,
                              dim_coords_and_dims=coords,
                              attributes=cube.attributes)
    logger.trace(new_cube)
    return new_cube


def sysrun(cmd):
    """Run a shell command and return a completed process object.

    This wrapper enforces:
    - ``check=True`` so failures raise ``CalledProcessError``;
    - captured stdout/stderr for later logging/inspection; and
    - UTF-8 text decoding for convenient string handling.
    """
    return sp.run(cmd, check=True, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf8')


def load_config(file_path):
    """Dynamically import and return a Python config module from disk.

    Parameters
    ----------
    file_path : str | Path
        Path to a ``.py`` configuration file.

    Returns
    -------
    module
        Imported module object whose attributes can be used as config values.

    Notes
    -----
    The module is inserted into ``sys.modules`` so normal import machinery can
    resolve references consistently if the module is imported again later.
    """
    path = Path(file_path)
    # Use filename stem as synthetic module name (e.g., "hk26_config").
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
    """Return an Iris constraint that keeps cubes with an exact dim order.

    Parameters
    ----------
    *dims : str
        Expected sequence of dimension coordinate names.

    Returns
    -------
    iris.Constraint
        Constraint with a cube function that compares ``cube.dim_coords`` names
        exactly (both names and order) against ``dims``.
    """

    def dim_filter(cube):
        # Extract dimension coordinate names in order.
        cube_dims = tuple([c.name() for c in cube.dim_coords])
        # Require exact match; no subset/superset matching.
        return cube_dims == dims

    return iris.Constraint(cube_func=dim_filter)


def cube_cell_method_is_not_empty(cube):
    """Return True when a cube has one or more Iris cell methods."""
    return cube.cell_methods != tuple()


def cube_cell_method_is_empty(cube):
    """Return True when a cube has no Iris cell methods."""
    return cube.cell_methods == tuple()


def invert_cube_sign(cube):
    """Multiply cube data by -1 in place and return the same cube."""
    cube.data = -1 * cube.data
    return cube


def check_cube_time_length(cube):
    """Trim leading time step when cube has an initial length of 13.

    This is a targeted normalization for known inputs where the first cube in a
    sequence may contain an extra leading timestep.
    """
    # Shorten cube if it has length 13 (applies to first cube only I think).
    if cube.shape[0] == 13:
        cube = cube[1:]
    return cube


def open_remote_dataset(config, sim, freq, zoom, on_jasmin=False):
    """Open a remote UM HEALPix Zarr dataset and attach HEALPix coordinates.

    Parameters
    ----------
    config : module/object
        Configuration object expected to expose ``deploy`` and ``output_vn``.
    sim : str
        Simulation identifier used in the object-store path.
    freq : str
        Output frequency token used in filename pattern ``um.{freq}.hp_z...``.
    zoom : int | str
        HEALPix zoom level used in the dataset name.
    on_jasmin : bool, default=False
        If True, use internal JASMIN endpoint over HTTP; otherwise use external
        endpoint over HTTPS.

    Returns
    -------
    xarray.Dataset
        Opened dataset with HEALPix coordinate metadata attached.
    """
    # Select object-store endpoint depending on execution environment.
    if on_jasmin:
        protocol = 'http'
        baseurl = 'hackathon-o.s3.jc.rl.ac.uk'
    else:
        protocol = 'https'
        baseurl = 'hackathon-o.s3-ext.jc.rl.ac.uk'
    # Construct canonical dataset URL in the project storage layout.
    url = f'{protocol}://{baseurl}/sim-data/{config.deploy}/{config.output_vn}/{sim}/um.{freq}.hp_z{zoom}.zarr/'
    print(url)

    # Open remote Zarr through xarray and attach derived HEALPix coordinates
    # used by downstream analysis/plotting code.
    ds = xr.open_dataset(url, engine='zarr')
    ds = ds.pipe(egh.attach_coords)
    return ds
