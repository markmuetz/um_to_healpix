import math as maths

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

import easygems.healpix as egh


def get_plot_kwargs(da):
    if abs(da.max() + da.min()) / (da.max() - da.min()) < 0.5:
        # data looks like it needs a diverging cmap.
        # figure out some nice bounds.
        pl, pu = np.percentile(da.values[~np.isnan(da.values)], [2, 98])
        vmax = np.abs([pl, pu]).max()
        kwargs = dict(
            cmap='bwr',
            vmin=-vmax,
            vmax=vmax,
        )
    else:
        kwargs = {}
    return kwargs


def plot_all_fields(ds_plot):
    """Plot all fields for a given dataset. Assumes that each field is 2D - i.e. sel(time=..., [pressure=...]) has been applied"""
    zoom = int(np.log2(ds_plot.crs.attrs['healpix_nside']))
    projection = ccrs.Robinson(central_longitude=0)
    das = {}
    for name, da in ds_plot.data_vars.items():
        if name == 'mrso':
            das.update({name + str(i): da[i] for i in range(len(da.depth))})
        else:
            das[name] = da

    rows = maths.ceil(len(das) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(30, rows * 20 / 6), subplot_kw={'projection': projection},
                             layout='constrained')
    if 'pressure' in ds_plot.coords:
        plt.suptitle(f'{ds_plot.simulation} z{zoom} @{float(ds_plot.pressure)}hPa')
    else:
        plt.suptitle(f'{ds_plot.simulation} z{zoom}')

    for ax, (name, da) in zip(axes.flatten(), das.items()):
        time = pd.Timestamp(ds_plot.time.values.item())

        kwargs = get_plot_kwargs(da)

        ax.set_title(f'time: {time} - {name}')
        ax.set_global()
        im = egh.healpix_show(da, ax=ax, **kwargs)
        long_name = da.long_name

        plt.colorbar(im, label=f'{long_name} ({da.attrs.get("units", "-")})')
        ax.coastlines()

def plot_timeseries(das):
    for zoom, da in das.items():
        logger.debug('plot_precip', zoom)
        plt.plot(da.mean(dim='cell'), label=f'z{zoom}')
    plt.legend()
    plt.xlabel('hour')
    plt.ylabel(f'{da.long_name} ({da.attrs.get("units", "-")})')

def plot_field_for_times(da, callback=None):
    projection = ccrs.Robinson(central_longitude=0)
    vmin = da.min()
    vmax = da.max()

    for tidx in range(len(da.time)):
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), subplot_kw={'projection': projection},
                               layout='constrained')
        print(tidx)
        time = pd.Timestamp(da.time.values[tidx])

        kwargs = get_plot_kwargs(da)
        kwargs['vmin'] = vmin
        kwargs['vmax'] = vmax

        ax.set_title(f'time: {time} - {da.name}')
        ax.set_global()
        im = egh.healpix_show(da.isel(time=tidx), ax=ax, **kwargs)
        long_name = da.long_name

        plt.colorbar(im, label=f'{long_name} ({da.attrs.get("units", "-")})')
        ax.coastlines()
        if callback:
            callback(tidx)



def plot_zonal_mean(das, bins=np.linspace(-90, 90, 1801)):
    bin_mids = (bins[:-1] + bins[1:]) / 2

    plt.figure(layout='constrained')
    for da in das:
        zoom = int(np.log2(da.crs.attrs['healpix_nside']))
        zonal_mean_da = da.mean(dim='time').groupby_bins('lat', bins).mean()
        plt.plot(bin_mids, zonal_mean_da, label=f'z{zoom}')

    plt.legend()

    plt.xlim((-90, 90))
    plt.xlabel('lat')
    plt.ylabel(f'{da.long_name} ({da.attrs.get("units", "-")})')

