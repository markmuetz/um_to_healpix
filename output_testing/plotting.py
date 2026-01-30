import math as maths

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    rows = maths.ceil(len(ds_plot.data_vars) / 4)
    fig, axes = plt.subplots(rows, 4, figsize=(30, rows * 20 / 6), subplot_kw={'projection': projection},
                             layout='constrained')
    if 'pressure' in ds_plot.coords:
        plt.suptitle(f'{ds_plot.simulation} z{zoom} @{float(ds_plot.pressure)}hPa')
    else:
        plt.suptitle(f'{ds_plot.simulation} z{zoom}')

    for ax, (name, da) in zip(axes.flatten(), ds_plot.data_vars.items()):
        time = pd.Timestamp(ds_plot.time.values.item())

        kwargs = get_plot_kwargs(da)

        ax.set_title(f'time: {time} - {name}')
        ax.set_global()
        im = egh.healpix_show(da, ax=ax, **kwargs)
        long_name = da.long_name

        plt.colorbar(im, label=f'{long_name} ({da.attrs.get("units", "-")})')
        ax.coastlines()
    plt.show()

def plot_field_for_times(da, field):
    projection = ccrs.Robinson(central_longitude=0)

    for tidx in range(len(da.time)):
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), subplot_kw={'projection': projection},
                               layout='constrained')
        print(tidx)
        time = pd.Timestamp(da.time.values[tidx])

        kwargs = get_plot_kwargs(da)

        ax.set_title(f'time: {time} - {field}')
        ax.set_global()
        im = egh.healpix_show(da.isel(time=tidx), ax=ax, **kwargs)
        long_name = da.long_name

        plt.colorbar(im, label=f'{long_name} ({da.attrs.get("units", "-")})')
        ax.coastlines()
        plt.show()


