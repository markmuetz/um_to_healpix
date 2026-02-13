"""Utilities for coarsening HEALPix Zarr datasets by one zoom level.

This module provides the full coarsening path used by the pipeline:
- map regional cell domains to temporary global indexing,
- coarsen source data by grouping each parent cell's 4 children,
- optionally map coarsened output back to a regional target domain,
- derive/propagate metadata and weights, and
- write each variable into a target Zarr store with retry logic.

The implementation is intentionally explicit to keep control over coordinate
alignment and write regions, which are both easy to get wrong with HEALPix cell
subsets and partial writes.
"""
import asyncio

import numpy as np
import xarray as xr
from loguru import logger

from .util import async_da_to_zarr_with_retries

def nan_weight(arr, axis):
    """Return fraction of non-NaN child values per coarsening group.

    Parameters
    ----------
    arr : numpy.ndarray
        Input block provided by ``xarray.coarsen(...).reduce(...)``.
    axis : int | tuple[int]
        Axis/axes over which the reduction is performed.

    Returns
    -------
    numpy.ndarray
        ``float32`` weights in [0, 1], computed as
        ``count_non_nan / 4`` for HEALPix 4-to-1 coarsening.

    Notes
    -----
    For this pipeline, each target HEALPix cell is formed from exactly 4 source
    children, so dividing by 4 gives the valid-data fraction.
    """
    # Invert np.isnan to get valid-data mask, count valid values per block, and
    # normalize by 4 because each parent has 4 child cells at the next zoom.
    return ((~np.isnan(arr)).sum(axis=axis) / 4).astype(np.float32)


def map_regional_to_global(_da, src_zoom, dim):
    """Map a regional DataArray onto a full global HEALPix cell axis.

    Parameters
    ----------
    _da : xarray.DataArray
        Regional source data whose last dimension is ``cell``.
    src_zoom : int
        Source HEALPix zoom level. Used to infer global cell count
        ``12 * 4**src_zoom``.
    dim : {'2d', '3d'}
        Dimensionality marker used to reconstruct expected dimension names.

    Returns
    -------
    xarray.DataArray
        Global-shaped DataArray with NaN-filled cells outside the regional mask.
    """
    logger.debug(_da.name)

    # Keep all existing coordinates except 'cell'; we will rebuild cell globally.
    coords = {
        n: c for n, c in _da.coords.items()
        if n != 'cell'
    }
    # HEALPix has 12 base faces and 4-way refinement per zoom level.
    coords['cell'] = np.arange(12 * 4 ** src_zoom)

    # Reconstruct canonical dims/shape so assignment is deterministic.
    if _da.name == 'weights':
        dims = ['cell']
        shape = [len(coords['cell'])]
    else:
        if dim == '2d':
            dims = ['time', 'cell']
        elif dim == '3d':
            dims = ['time', 'pressure', 'cell']
        # dims = [d for d in coords if d != 'crs']
        shape = list(_da.shape[:-1]) + [len(coords['cell'])]

    # Would be nice, but it's actually quite a bit slower?
    # dummies = dask.array.zeros(shape, dtype=np.float32)
    # da_new = xr.DataArray(dummies, dims=dims, name=_da.name, coords=coords)
    # Pre-fill with NaN to mark cells outside the regional subset.
    da_new = xr.DataArray(np.full(shape, np.nan, np.float32), dims=dims, name=_da.name, coords=coords)
    # Dangerously wrong! Will silently ignore this.
    # da_new.isel(cell=_da.cell).values = _da.values

    # This on the other hand works perfectly, because .loc allows assignment.
    # Coordinate-based assignment ensures values are written into matching cells.
    da_new.loc[dict(cell=_da.cell)] = _da
    return da_new


def map_global_to_regional(_da, src_ds_time_slice, tgt_ds_store, dim):
    """Map a global DataArray back onto the regional target cell subset.

    Parameters
    ----------
    _da : xarray.DataArray
        Global coarsened DataArray with full cell axis.
    src_ds_time_slice : xarray.Dataset
        Source time-sliced dataset (used only for shape reference of non-weight
        variables).
    tgt_ds_store : xarray.Dataset
        Opened target store that provides the regional ``cell`` coordinate.
    dim : {'2d', '3d'}
        Dimensionality marker used to rebuild dims.

    Returns
    -------
    xarray.DataArray
        DataArray restricted to target regional cells.
    """
    logger.debug(_da.name)

    # Preserve non-cell coordinates and replace cell with regional target cells.
    coords = {
        n: c for n, c in _da.coords.items()
        if n != 'cell'
    }
    coords['cell'] = tgt_ds_store.cell

    # Recreate expected shape for the regional array.
    if _da.name == 'weights':
        dims = ['cell']
        shape = [len(coords['cell'])]
    else:
        if dim == '2d':
            dims = ['time', 'cell']
        elif dim == '3d':
            dims = ['time', 'pressure', 'cell']
        da = list(src_ds_time_slice.data_vars.values())[0]
        shape = list(da.shape[:-1]) + [len(coords['cell'])]

    # Allocate output and then gather by integer cell indices from global array.
    da_new = xr.DataArray(np.full(shape, np.nan, np.float32), dims=dims, name=_da.name, coords=coords)
    da_new.values = _da.isel(cell=tgt_ds_store.cell.values)
    return da_new


def calc_tgt_weights(src_ds, tgt_ds, tgt_zoom, dim):
    """Calculate target-level valid-data weights for regional datasets.

    Parameters
    ----------
    src_ds : xarray.Dataset
        Source dataset at ``src_zoom = tgt_zoom + 1``.
    tgt_ds : xarray.Dataset
        Target dataset used to define regional target cell selection.
    tgt_zoom : int
        Target HEALPix zoom level.
    dim : {'2d', '3d'}
        Dimensionality marker used to select a representative slice.

    Returns
    -------
    xarray.DataArray
        ``weights`` array aligned to ``tgt_ds.cell``.

    Notes
    -----
    Weights are estimated from a single representative variable/slice because
    the regional mask topology is shared across variables.
    """
    # It is enough to calc the weights for one field, and store.
    if dim == '2d':
        # For 2D variables, use one time slice.
        src_da = list(src_ds.data_vars.values())[0].isel(time=0)
    elif dim == '3d':
        # For 3D variables, fix both time and pressure for a representative slice.
        src_da = list(src_ds.data_vars.values())[0].isel(time=0, pressure=0)

    # Coarsen by 4 cells and compute fraction of valid children in each parent.
    weights = src_da.coarsen(cell=4).reduce(nan_weight).compute()
    cells = np.arange(12 * 4 ** tgt_zoom)
    coords = {'cell': cells}

    # Use a global weights DataArray to convert between the src domain and the tgt domain.
    weights_global = xr.DataArray(np.full(12 * 4 ** tgt_zoom, np.nan, np.float32), dims=['cell'], coords=coords)
    # I'm found it difficult to figure out how to use cells to map into weights_global
    # But this works.
    # Derive parent cell ids from source child ids using 4-to-1 HEALPix grouping.
    tgt_cell_from_src = ((src_ds.cell.values.reshape(-1, 4).mean(axis=-1) - 1.5) / 4).astype(int)
    weights_global.loc[dict(cell=tgt_cell_from_src)] = weights.values

    # Restrict back to the regional target cell list.
    tgt_weights = weights_global.isel(cell=tgt_ds.cell)
    logger.debug(float(np.isnan(tgt_weights).sum().values))
    return tgt_weights


def coarsen_healpix_zarr_region(src_ds, tgt_store, tgt_zoom, dim, start_idx, end_idx, chunks, regional=False):
    """Coarsen a time-region from source HEALPix Zarr into a target Zarr store.

    Parameters
    ----------
    src_ds : xarray.Dataset
        Source dataset at zoom ``tgt_zoom + 1``.
    tgt_store : str | MutableMapping | fsspec-compatible store
        Destination Zarr store path/handle.
    tgt_zoom : int
        Target HEALPix zoom level.
    dim : {'2d', '3d'}
        Dataset dimensionality mode used for chunk/region shape logic.
    start_idx, end_idx : int
        Time index boundaries (Python slice semantics).
    chunks : dict[int, tuple[int, ...]]
        Chunk specification keyed by zoom level.
    regional : bool, default=False
        If True, perform regional<->global mapping before/after coarsening.

    Notes
    -----
    - Coarsening is always 4:1 on the ``cell`` axis.
    - Writes are done variable-by-variable with retry logic to handle transient
      object-store and filesystem failures.
    """
    # Select time span for this write chunk.
    time_slice = slice(start_idx, end_idx)
    # Source is one HEALPix level finer than target.
    src_zoom = tgt_zoom + 1
    logger.info(f'Coarsen to {tgt_zoom}, time_slice={time_slice}')

    # Target chunking metadata expected by downstream readers.
    tgt_chunks = chunks[tgt_zoom]

    # Restrict source dataset to requested time region.
    src_ds_time_slice = src_ds.isel(time=time_slice)
    if regional:
        # For 3D regional runs, avoid carrying pre-existing weights through map,
        # then recompute at target resolution when needed.
        if dim == '3d' and 'weights' in src_ds.data_vars.keys():
            logger.debug('drop weights')
            src_ds_time_slice = src_ds_time_slice.drop_vars('weights')

        # Expand regional cells to global index space so coarsen grouping is valid.
        src_ds_time_slice = src_ds_time_slice.map(map_regional_to_global, src_zoom=src_zoom, dim=dim)

    # To compute or not to compute...
    # Computing would force the download and loading into mem, but might stop contention for the resource?
    # tgt_ds = src_ds_time_slice.coarsen(cell=4).mean()
    logger.debug('compute tgt_ds')
    # Perform spatial coarsening over HEALPix cell groups.
    tgt_ds = src_ds_time_slice.coarsen(cell=4).mean().compute()

    logger.info('modify metadata')
    # Build preferred chunk metadata based on data dimensionality.
    if dim == '2d':
        preferred_chunks = {'time': tgt_chunks[0], 'cell': tgt_chunks[1]}
    else:
        preferred_chunks = {'time': tgt_chunks[0], 'pressure': tgt_chunks[1], 'cell': tgt_chunks[2]}

    # Stamp per-variable zoom/chunk metadata so written arrays are self-describing.
    for da in tgt_ds.data_vars.values():
        da.attrs['healpix_zoom'] = tgt_zoom
        da.encoding['chunks'] = tgt_chunks
        da.encoding['preferred_chunks'] = preferred_chunks

    if regional:
        # Open target store to read regional cell coordinate definition.
        zarr_chunks = {'time': chunks[tgt_zoom][0], 'cell': -1}
        tgt_ds_store = xr.open_zarr(tgt_store, chunks=zarr_chunks)

        # Reduce global coarsened data back to target regional cell subset.
        tgt_ds = tgt_ds.map(map_global_to_regional, src_ds_time_slice=src_ds_time_slice, tgt_ds_store=tgt_ds_store, dim=dim)

        # Add weights once, only for the first processed time block, unless
        # source already carried weights.
        if (src_ds_time_slice.time[0] == src_ds.time[0]) and 'weights' not in src_ds.data_vars:
            tgt_weights = calc_tgt_weights(src_ds, tgt_ds, tgt_zoom, dim)
            tgt_ds['weights'] = tgt_weights
            logger.debug(float(np.isnan(tgt_ds.weights).sum().values))

    # 'crs' is coordinate/reference metadata that should not be rewritten here.
    tgt_ds = tgt_ds.drop_vars('crs')

    # Build Zarr write region based on variable dimensionality.
    if dim == '2d':
        region = {'time': time_slice, 'cell': slice(None)}
    elif dim == '3d':
        region = {'time': time_slice, 'pressure': slice(None), 'cell': slice(None)}
    else:
        raise ValueError(f'dim {dim} is not supported')

    # Write each variable independently. This keeps failures isolated and allows
    # retry behavior per variable.
    for da in tgt_ds.data_vars.values():
        if np.isnan(da.values).all():
            # This can happen for e.g. the first timestep so don't raise an exception.
            logger.error(f'ERROR: ALL VALUES ARE NAN FOR {da.name}')
        else:
            logger.debug('values not all nan')

        # Static fields are not rewritten during rolling time-block updates.
        if da.name in ['orog', 'sftlf']:
            continue

        logger.debug(f'  writing {da.name}')
        if da.name == 'weights':
            # Weights are written once for the initial block only.
            if src_ds_time_slice.time[0] == src_ds.time[0]:
                if dim == '2d':
                    # 2D weights are 1D over cell only.
                    region = {'cell': slice(None)}
                    asyncio.run(
                        async_da_to_zarr_with_retries(da.chunk({'cell': preferred_chunks['cell']}), tgt_store, region))
                elif dim == '3d':
                    # TODO: still causes error due to dims mismatch.
                    logger.warning('Not writing weights for 3D data')
                    # region = {'pressure': slice(None), 'cell': slice(None)}
                    # asyncio.run(
                    #     async_da_to_zarr_with_retries(da.chunk({'cell': preferred_chunks['cell']}), tgt_store, region))
            continue

        # Main variable write path with async retry wrapper.
        asyncio.run(
            async_da_to_zarr_with_retries(da.chunk({'cell': preferred_chunks['cell']}), tgt_store, region))
