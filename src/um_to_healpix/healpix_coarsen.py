"""Includes all the code for coarsening from one healpix level to the next lower."""
import asyncio

import numpy as np
import xarray as xr
from loguru import logger

from .util import async_da_to_zarr_with_retries

# TODO: docstrings.


def nan_weight(arr, axis):
    return ((~np.isnan(arr)).sum(axis=axis) / 4).astype(np.float32)


def map_regional_to_global(_da, src_zoom, dim):
    logger.debug(_da.name)
    coords = {
        n: c for n, c in _da.coords.items()
        if n != 'healpix_index'
    }
    coords['healpix_index'] = np.arange(12 * 4 ** src_zoom)

    if _da.name == 'weights':
        dims = ['healpix_index']
        shape = [len(coords['healpix_index'])]
    else:
        if dim == '2d':
            dims = ['time', 'healpix_index']
        elif dim == '3d':
            dims = ['time', 'pressure', 'healpix_index']
        # dims = [d for d in coords if d != 'crs']
        shape = list(_da.shape[:-1]) + [len(coords['healpix_index'])]
    # Would be nice, but it's actually quite a bit slower?
    # dummies = dask.array.zeros(shape, dtype=np.float32)
    # da_new = xr.DataArray(dummies, dims=dims, name=_da.name, coords=coords)
    da_new = xr.DataArray(np.full(shape, np.nan, np.float32), dims=dims, name=_da.name, coords=coords)
    # Dangerously wrong! Will silently ignore this.
    # da_new.isel(healpix_index=_da.healpix_index).values = _da.values

    # This on the other hand works perfectly, because .loc allows assignment.
    da_new.loc[dict(healpix_index=_da.healpix_index)] = _da
    return da_new


def map_global_to_regional(_da, src_ds_time_slice, tgt_ds_store, dim):
    logger.debug(_da.name)
    coords = {
        n: c for n, c in _da.coords.items()
        if n != 'healpix_index'
    }
    coords['healpix_index'] = tgt_ds_store.healpix_index
    if _da.name == 'weights':
        dims = ['healpix_index']
        shape = [len(coords['healpix_index'])]
    else:
        if dim == '2d':
            dims = ['time', 'healpix_index']
        elif dim == '3d':
            dims = ['time', 'pressure', 'healpix_index']
        da = list(src_ds_time_slice.data_vars.values())[0]
        shape = list(da.shape[:-1]) + [len(coords['healpix_index'])]

    da_new = xr.DataArray(np.full(shape, np.nan, np.float32), dims=dims, name=_da.name, coords=coords)
    da_new.values = _da.isel(healpix_index=tgt_ds_store.healpix_index.values)
    return da_new


def calc_tgt_weights(src_ds, tgt_ds, tgt_zoom, dim):
    # It is enough to calc the weights for one field, and store.
    if dim == '2d':
        src_da = list(src_ds.data_vars.values())[0].isel(time=0)
    elif dim == '3d':
        src_da = list(src_ds.data_vars.values())[0].isel(time=0, pressure=0)
    weights = src_da.coarsen(healpix_index=4).reduce(nan_weight).compute()
    cells = np.arange(12 * 4 ** tgt_zoom)
    coords = {'healpix_index': cells}
    # Use a global weights DataArray to convert between the src domain and the tgt domain.
    weights_global = xr.DataArray(np.full(12 * 4 ** tgt_zoom, np.nan, np.float32), dims=['healpix_index'], coords=coords)
    # I'm found it difficult to figure out how to use cells to map into weights_global
    # But this works.
    tgt_cell_from_src = ((src_ds.healpix_index.values.reshape(-1, 4).mean(axis=-1) - 1.5) / 4).astype(int)
    weights_global.loc[dict(healpix_index=tgt_cell_from_src)] = weights.values
    tgt_weights = weights_global.isel(healpix_index=tgt_ds.healpix_index)
    logger.debug(float(np.isnan(tgt_weights).sum().values))
    return tgt_weights


def coarsen_healpix_zarr_region(src_ds, tgt_store, tgt_zoom, dim, start_idx, end_idx, chunks, regional=False):
    time_slice = slice(start_idx, end_idx)
    src_zoom = tgt_zoom + 1
    logger.info(f'Coarsen to {tgt_zoom}, time_slice={time_slice}')

    tgt_chunks = chunks[tgt_zoom]

    src_ds_time_slice = src_ds.isel(time=time_slice)
    src_ds_time_slice = src_ds_time_slice.chunk({'time': -1, 'healpix_index': -1})

    if regional:
        if dim == '3d' and 'weights' in src_ds.data_vars.keys():
            logger.debug('drop weights')
            src_ds_time_slice = src_ds_time_slice.drop_vars('weights')
        src_ds_time_slice = src_ds_time_slice.map(map_regional_to_global, src_zoom=src_zoom, dim=dim)

    # To compute or not to compute...
    # Computing would force the download and loading into mem, but might stop contention for the resource?
    # tgt_ds = src_ds_time_slice.coarsen(healpix_index=4).mean()
    logger.debug('compute tgt_ds')
    tgt_ds = src_ds_time_slice.coarsen(healpix_index=4).mean().compute()

    logger.info('modify metadata')
    if dim == '2d':
        preferred_chunks = {'time': tgt_chunks[0], 'healpix_index': tgt_chunks[1]}
    else:
        preferred_chunks = {'time': tgt_chunks[0], 'pressure': tgt_chunks[1], 'healpix_index': tgt_chunks[2]}
    for da in tgt_ds.data_vars.values():
        da.attrs['healpix_zoom'] = tgt_zoom
        da.encoding['chunks'] = tgt_chunks
        da.encoding['preferred_chunks'] = preferred_chunks

    if regional:
        zarr_chunks = {'time': chunks[tgt_zoom][0], 'healpix_index': -1}
        tgt_ds_store = xr.open_zarr(tgt_store, chunks=zarr_chunks)
        tgt_ds = tgt_ds.map(map_global_to_regional, src_ds_time_slice=src_ds_time_slice, tgt_ds_store=tgt_ds_store, dim=dim)
        if (src_ds_time_slice.time[0] == src_ds.time[0]) and 'weights' not in src_ds.data_vars:
            tgt_weights = calc_tgt_weights(src_ds, tgt_ds, tgt_zoom, dim)
            tgt_ds['weights'] = tgt_weights
            logger.debug(float(np.isnan(tgt_ds.weights).sum().values))

    tgt_ds = tgt_ds.drop_vars('crs')

    if dim == '2d':
        region = {'time': time_slice, 'healpix_index': slice(None)}
    elif dim == '3d':
        region = {'time': time_slice, 'pressure': slice(None), 'healpix_index': slice(None)}
    else:
        raise ValueError(f'dim {dim} is not supported')
    for da in tgt_ds.data_vars.values():
        if np.isnan(da.values).all():
            # This can happen for e.g. the first timestep so don't raise an exception.
            logger.error(f'ERROR: ALL VALUES ARE NAN FOR {da.name}')
        else:
            logger.debug('values not all nan')
        if da.name in ['orog', 'sftlf']:
            continue
        logger.debug(f'  writing {da.name}')
        if da.name == 'weights':
            if src_ds_time_slice.time[0] == src_ds.time[0]:
                if dim == '2d':
                    wregion = {'healpix_index': slice(None)}
                    asyncio.run(
                        async_da_to_zarr_with_retries(da.chunk({'healpix_index': preferred_chunks['healpix_index']}), tgt_store, wregion))
                elif dim == '3d':
                    # TODO: still causes error due to dims mismatch.
                    logger.warning('Not writing weights for 3D data')
                    # region = {'pressure': slice(None), 'healpix_index': slice(None)}
                    # asyncio.run(
                    #     async_da_to_zarr_with_retries(da.chunk({'healpix_index': preferred_chunks['healpix_index']}), tgt_store, region))
            continue
        if da.name == 'mrsol':
            dregion = {'time': time_slice, 'depth': slice(None), 'healpix_index': slice(None)}
            asyncio.run(
                async_da_to_zarr_with_retries(da.chunk({'healpix_index': preferred_chunks['healpix_index']}), tgt_store, dregion))
        else:
            asyncio.run(
                async_da_to_zarr_with_retries(da.chunk({'healpix_index': preferred_chunks['healpix_index']}), tgt_store, region))
