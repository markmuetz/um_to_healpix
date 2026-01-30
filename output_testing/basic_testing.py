# %%
import math as maths

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import easygems.healpix as egh

from um_to_healpix.util import load_config

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
    return ds


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
        time = pd.Timestamp(ds_plot.time.values[0])

        kwargs = get_plot_kwargs(da)

        ax.set_title(f'time: {time} - {name}')
        ax.set_global()
        im = egh.healpix_show(da, ax=ax, **kwargs)
        long_name = da.long_name

        plt.colorbar(im, label=f'{long_name} ({da.attrs.get("units", "-")})')
        ax.coastlines()


def plot_field_for_times(da, field):
    projection = ccrs.Robinson(central_longitude=0)

    for tidx in range(len(da.time)):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10), subplot_kw={'projection': projection},
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


if __name__ == '__main__':
    # %%
    config = load_config('config/hk26_config.py')
    sim = 'glm.n2560_RAL3p3.tuned'
    freq = 'PT1H'  # 2D
    # freq = 'PT3H'  # 3D
    zoom = 10  # Only zoom with any data in so far.
    on_jasmin = False

    # %% Load data.
    ds = open_remote_dataset(config, sim, freq, zoom, on_jasmin)
    da = ds['tas'].sel(time=slice(pd.Timestamp('2020-01-20 00:00:00'), pd.Timestamp('2020-01-20 05:00:00'))).compute()
    ds_plot = ds.isel(time=4).compute()

    # %%
    plot_field_for_times(da, 'tas')
    # %%
    # plot_all_fields(ds_plot)
