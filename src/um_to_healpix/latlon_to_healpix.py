"""Regridding utilities from regular lat/lon fields to HEALPix nested indexing.

This module provides:
- helper functions for HEALPix cell/domain selection,
- generation of Delaunay interpolation weights, and
- a ``LatLon2HealpixRegridder`` class that applies either
    ``easygems_delaunay`` or ``earth2grid`` regridding.

The core workflow is:
1. optionally add a cyclic longitude column,
2. derive source/target coordinates,
3. interpolate for each non-lat/lon slice,
4. return an output DataArray with a ``cell`` dimension.
"""
from itertools import product
from pathlib import Path

import easygems.healpix as egh
import easygems.remap as egr
import healpix as hp
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
from loguru import logger


def _xr_add_cyclic_point(da, lonname):
    """Add a cyclic longitude column to an xarray DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with a longitude dimension.
    lonname : str
        Name of longitude coordinate/dimension in ``da``.

    Returns
    -------
    xarray.DataArray
        New DataArray with one extra longitude column and wrapped coordinate,
        preserving original dims/attrs/name.

    Notes
    -----
    Adding a cyclic point helps ensure the source convex hull covers wraparound
    longitudes, reducing interpolation-edge NaNs near 0/360.
    """

    # Use add_cyclic_point to interpolate input data
    # Identify numeric axis index for lon dimension as expected by cartopy util.
    lon_idx = da.dims.index(lonname)
    # Returns wrapped data and wrapped longitude coordinate.
    wrap_data, wrap_lon = add_cyclic_point(da.values, coord=da[lonname], axis=lon_idx)
    # Rebuild coords with updated longitude only; keep all other coordinates.
    coords = {n: c for n, c in da.coords.items() if n != lonname}
    coords[lonname] = wrap_lon

    # Generate output DataArray with new data but same structure as input
    daout = xr.DataArray(data=wrap_data,
                         name=da.name,
                         coords=coords,
                         dims=da.dims,
                         attrs=da.attrs)
    return daout


def get_limited_healpix(extent, zoom, chunksize):
    """Return regional HEALPix lon/lat/cell indices aligned to full chunks.

    Parameters
    ----------
    extent : tuple[float, float, float, float]
        Bounding box ``(minlon, maxlon, minlat, maxlat)``.
    zoom : int
        HEALPix zoom/order.
    chunksize : int
        Desired cell chunk size used by ``easygems.healpix.get_full_chunks``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(hp_lon_chunked, hp_lat_chunked, ichunk)`` where ``ichunk`` indexes
        full global-cell IDs limited to extent and expanded to chunk boundaries.
    """
    # https://gitlab.dkrz.de/-/snippets/81
    # First find cells inside requested region.
    hp_lon, hp_lat, icell = get_regional_cell_idx(extent, zoom)
    # Then expand/align to full chunk boundaries used in storage layout.
    ichunk = egh.get_full_chunks(icell, chunksize=chunksize)

    return hp_lon[ichunk], hp_lat[ichunk], ichunk


def get_extent(da, lonname, latname):
    """Extract rectangular lon/lat extent from coordinate endpoints.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with latitude and longitude coordinates.
    lonname, latname : str
        Coordinate names.

    Returns
    -------
    tuple[float, float, float, float]
        ``(minlon, maxlon, minlat, maxlat)`` based on first/last coord values.
    """
    # Assumes coordinates are ordered; uses edge values directly.
    minlon, maxlon = da[lonname].values[[0, -1]]
    minlat, maxlat = da[latname].values[[0, -1]]
    return minlon, maxlon, minlat, maxlat


def get_regional_cell_idx(extent, zoom):
    """Get global HEALPix cell indices whose centers fall within an extent.

    Parameters
    ----------
    extent : tuple[float, float, float, float]
        Bounding box ``(minlon, maxlon, minlat, maxlat)``.
    zoom : int
        HEALPix zoom/order.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(hp_lon, hp_lat, icell)`` where lon/lat are for all global cells at
        ``zoom`` and ``icell`` selects the subset inside ``extent``.

    Notes
    -----
    Longitude handling includes a >360° extent case used by regional domains
    crossing the dateline in a shifted coordinate convention.
    """
    # Convert zoom/order to nside and total number of pixels.
    nside = hp.order2nside(zoom)
    npix = hp.nside2npix(nside)

    # Compute lon/lat of every HEALPix cell center in NEST ordering.
    hp_lon, hp_lat = hp.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
    # Normalize to [0, 360) for consistent comparisons with source longitudes.
    hp_lon = hp_lon % 360
    if extent[1] > 360:
        # The Africa regional domain has extent[1] == max lon = 415.
        # Shift low longitudes to 360..415-like range so extent comparison works.
        hp_lon = np.where(hp_lon <= extent[1] - 360, hp_lon + 360, hp_lon)
    # hp_lon = (hp_lon + 180) % 360 - 180

    # Keep cell centers strictly inside the bounding box.
    icell, = np.where(
        (hp_lon > extent[0]) &
        (hp_lon < extent[1]) &
        (hp_lat > extent[2]) &
        (hp_lat < extent[3])
    )
    return hp_lon, hp_lat, icell


def gen_weights(da, weights_path, zoom=10, lonname='longitude', latname='latitude', add_cyclic=True, regional=False,
                regional_chunks=None):
    """Generate delaunay weights for regridding.

    Can use quite a lot of RAM: 30-40G for a UM N2560 conversion.

    Assumption is that da has lon: 0 to 360, lat: -90 to 90.
    It is important to make sure that the input domain contains the output domain, i.e. its convex hull is bigger.
    Input domain is defined by the lat/lon coords in da, output domain is defined by healpix zoom level and is
    roughly 0 to 360.
    This is to ensure that the interpolation can proceed for all points - if you end up with NaNs in your output
    it could be because of this, and it might be necessary to add a cyclic point to the input domain.

    Parameters:
        da (xr.DataArray): input data array with lat/lon coords to use.
        weights_path (str): output path for serialized weights NetCDF.
        zoom (int): desired zoom level.
        lonname (str): name of longitude coord.
        latname (str): name of latitude coord.
        add_cyclic (bool): whether to add cyclic points.
        regional (bool): if True, generate weights only for a regional target.
        regional_chunks: currently unused placeholder for regional chunk logic.
    """
    # Normalize and prepare output destination.
    weights_path = Path(weights_path)
    # Safety check: avoid overwriting an existing weights file by accident.
    assert not weights_path.exists(), f'Weights file {weights_path} already exists'
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive total target HEALPix cells for this zoom.
    nside = hp.order2nside(zoom)
    npix = hp.nside2npix(nside)

    # Expand input domain by one in the lon dim.
    if add_cyclic:
        logger.debug('adding cyclic column')
        logger.trace(da[lonname])
        da = _xr_add_cyclic_point(da, lonname)
        logger.trace(da[lonname])

    # Full global target cell-center coordinates.
    hp_lon, hp_lat = hp.pix2ang(nside=nside, ipix=np.arange(npix), lonlat=True, nest=True)
    if regional:
        # TODO: or ichunk?
        # Reduce target points to those in source-data extent.
        hp_lon, hp_lat, icell = get_regional_cell_idx(get_extent(da, lonname, latname), zoom)
        hp_lon = hp_lon[icell]
        hp_lat = hp_lat[icell]
    else:
        # This was in code that I copied the function from but I think I can leave it out.
        # hp_lon += 360 / (4 * nside) / 4  # shift quarter-width
        hp_lon = hp_lon % 360  # [0, 360)
        # Apply a 360 degree offset. This ensures that all hp_lon are within da[lonname].
        hp_lon[hp_lon == 0] = 360

    # Flatten structured lat/lon source grid into a single interpolation axis.
    da_flat = da.stack(cell=(lonname, latname))

    logger.info('computing weights')
    # Build Delaunay interpolation weights from source points -> target points.
    weights = egr.compute_weights_delaunay((da_flat[lonname].values, da_flat[latname].values), (hp_lon, hp_lat))
    logger.debug(weights)
    # Persist weights for reuse by the regridder to avoid recomputation.
    weights.to_netcdf(weights_path)
    logger.info(f'saved weights to {weights_path}')


class LatLon2HealpixRegridder:
    """Regrid (UM) lat/lon .pp files to healpix .nc"""

    def __init__(self, weights_path, method='easygems_delaunay', zoom_level=10, add_cyclic=True, regional=False,
                 regional_chunks=None):
        """Initialize regridder configuration and load required resources.

        Parameters
        ----------
        weights_path : str | Path
            Path to precomputed Delaunay weights (required for
            ``method='easygems_delaunay'``).
        method : {'easygems_delaunay', 'earth2grid'}, default='easygems_delaunay'
            Regridding backend implementation.
        zoom_level : int, default=10
            Target HEALPix zoom/order.
        add_cyclic : bool, default=True
            Whether to add cyclic longitude point before regridding.
        regional : bool, default=False
            Whether to output only a regional subset of HEALPix cells.
        regional_chunks : int | None
            Chunk size used to expand regional cells to full chunk-aligned sets.
        """
        if method not in ['easygems_delaunay', 'earth2grid']:
            raise ValueError('method must be either easygems_delaunay or earth2grid')
        # Store user-selected runtime configuration.
        self.method = method
        self.zoom_level = zoom_level
        self.add_cyclic = add_cyclic
        self.weights_path = weights_path
        self.regional = regional
        if self.regional:
            self.regional_chunks = regional_chunks
        if method == 'easygems_delaunay':
            # Preload weights once for repeated slice-by-slice application.
            self.weights = xr.load_dataset(self.weights_path)

    def regrid(self, da, lonname, latname):
        """Regrid a DataArray from lat/lon to HEALPix cell dimension.

        Parameters
        ----------
        da : xarray.DataArray
            Input data array containing lat/lon dimensions plus optional leading
            dimensions (e.g., time, pressure).
        lonname, latname : str
            Longitude and latitude dimension names.

        Returns
        -------
        xarray.DataArray
            Regridded output with dims ``(non_latlon_dims..., cell)`` and
            metadata describing HEALPix mapping.
        """
        # Ensure wraparound continuity if configured.
        if self.add_cyclic:
            da = _xr_add_cyclic_point(da, lonname)

        # Preserve all non-spatial dimensions in output.
        reduced_dims = [d for d in da.dims if d not in [lonname, latname]]
        coords = {d: da[d] for d in reduced_dims}

        # This is the shape of the dataset without lat/lon.
        dim_shape = list(da.shape[:-2])
        # These are the ranges - can be used to iter over an idx that selects out each individual lat/lon field for
        # any number of dims by passing to product as product(*dim_ranges).
        dim_ranges = [range(s) for s in dim_shape]
        if self.regional:
            # Limit to region and chunk-aligned cells to match storage strategy.
            _, _, ichunk = get_limited_healpix(get_extent(da, lonname, latname), self.zoom_level, self.regional_chunks)
            cells = ichunk
            ncell = len(ichunk)
        else:
            # Full global HEALPix cell count at target zoom.
            ncell = 12 * 4 ** self.zoom_level
            cells = np.arange(ncell)

        # Allocate output as dense array; each idx in dim_ranges fills one row.
        regridded_data = np.zeros(dim_shape + [ncell])
        dim_len = {c: len(da[c]) for c in coords.keys()}
        logger.trace(f'  - {dim_len}')

        # Dispatch to requested backend implementation.
        if self.method == 'easygems_delaunay':
            self._regrid_easygems_delaunay(da, dim_ranges, regridded_data, lonname, latname)
        elif self.method == 'earth2grid':
            self._regrid_earth2grid(da, dim_ranges, regridded_data, lonname, latname)

        # Build final DataArray with original non-spatial coords plus cell ids.
        coords = {**coords, 'cell': cells}
        daout = xr.DataArray(
            regridded_data,
            name=da.name,
            dims=reduced_dims + ['cell'],
            coords=coords,
            attrs=da.attrs,
        )
        # Add metadata used downstream to identify HEALPix grid provenance.
        daout.attrs['grid_mapping'] = 'healpix_nested'
        daout.attrs['healpix_zoom'] = self.zoom_level
        daout.attrs['coarsened'] = 'False'
        daout.attrs['regrid_method'] = self.method
        return daout

    def _regrid_easygems_delaunay(self, da, dim_ranges, regridded_data, lonname, latname, regional=False):
        """Regrid using precomputed ``easygems.remap`` Delaunay weights.

        Parameters
        ----------
        da : xarray.DataArray
            Input array already prepared for regridding.
        dim_ranges : list[range]
            Index ranges for non-lat/lon dimensions.
        regridded_data : numpy.ndarray
            Preallocated output buffer to fill in-place.
        lonname, latname : str
            Spatial dimension names in ``da``.
        regional : bool, optional
            Unused parameter retained for backward compatibility.
        """
        # Flatten source lat/lon into a single point dimension expected by weights.
        da_flat = da.stack(cell=(lonname, latname))
        if self.regional:
            # Compute two related index sets:
            # - icell: valid cells strictly inside regional extent.
            # - ichunk: region expanded to full chunk boundaries.
            _, _, icell = get_regional_cell_idx(get_extent(da, lonname, latname), self.zoom_level)
            _, _, ichunk = get_limited_healpix(get_extent(da, lonname, latname), self.zoom_level, self.regional_chunks)
            # Allocate a reusable full-field buffer to avoid repeated allocations.
            field = np.full(12 * 4 ** self.zoom_level, np.nan, np.float32)
        else:
            icell = ichunk = field = None

        # Iterate over all non-spatial index combinations (or one empty tuple for 2D).
        for idx in product(*dim_ranges):
            logger.trace(f'    - {idx}')
            if self.regional:
                # This is a bit complicated.
                # icell: valid cells (not nan) based on extent
                # ichunk: valid cells based on extent and chunk.
                # both index a full field. Use this fact to convert between them.
                # We can get away with only allocating field once.
                # !!! NOTE: MASSIVELY FASTER IF I PASS IN .values !!!
                # Fill valid regional cells in full field, then extract chunked subset.
                field[icell] = egr.apply_weights(da_flat[idx].values, **self.weights)
                regridded_data[idx] = field[ichunk]
                # raise Exception()
            else:
                # Global case writes directly with no remapping indirection.
                regridded_data[idx] = egr.apply_weights(da_flat[idx].values, **self.weights)

    def _regrid_earth2grid(self, da, dim_ranges, regridded_data, lonname, latname):
        """Regrid using ``earth2grid`` + ``torch`` backend.

        Parameters
        ----------
        da : xarray.DataArray
            Input array prepared for regridding.
        dim_ranges : list[range]
            Index ranges for non-lat/lon dimensions.
        regridded_data : numpy.ndarray
            Preallocated output buffer to fill in-place.
        lonname, latname : str
            Spatial dimension names.
        """
        # I'm not assuming these will be installed.
        import earth2grid
        import torch

        # Build source lat/lon grid description for earth2grid.
        lat_lon_shape = (len(da[latname]), len(da[lonname]))
        src = earth2grid.latlon.equiangular_lat_lon_grid(*lat_lon_shape)

        # The y-dir is indexed in reverse for some reason.
        # Build a slice to invert latitude (for passing to regridding).
        # Keep all dims untouched except latitude, which is reversed.
        data_slice = [slice(None) if d != latname else slice(None, None, -1) for d in da.dims]
        target_data = da.values[*data_slice].copy().astype(np.double)

        # Note, you pass in PixelOrder.NEST here. .XY() (as in example) is equivalent to .RING.
        # Configure HEALPix target grid at required zoom/order.
        hpx = earth2grid.healpix.Grid(level=self.zoom_level, pixel_order=earth2grid.healpix.PixelOrder.NEST)
        regrid = earth2grid.get_regridder(src, hpx)

        # Regrid each non-spatial slice independently.
        for idx in product(*dim_ranges):
            logger.trace(f'    - {idx}')
            z_torch = torch.as_tensor(target_data[idx])
            z_hpx = regrid(z_torch)
            # if idx == () this still works (i.e. does nothing to regridded_data).
            regridded_data[idx] = z_hpx.numpy()
