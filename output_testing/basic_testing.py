# %% Autoreload magic.
from IPython import get_ipython
# Equivalent to but without the syntax problem.
# %load_ext autoreload
# %autoreload 2
# Get the active IPython instance
# Only run if we are actually in an interactive console
if ipy := get_ipython():
    ipy.run_line_magic('load_ext', 'autoreload')
    ipy.run_line_magic('autoreload', '2')

# %% Imports.
import pandas as pd
import xarray as xr

from um_to_healpix.util import load_config

from output_testing.plotting import plot_field_for_times, plot_all_fields

# %% Library code.
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


# %% Settings.
config = load_config('config/hk26_config.py')
sim = 'glm.n2560_RAL3p3.tuned'
freq = 'PT1H'  # 2D
# freq = 'PT3H'  # 3D
zoom = 10  # Only zoom with any data in so far.
on_jasmin = False

# %% Load data.
ds = open_remote_dataset(config, sim, freq, zoom, on_jasmin)
da = ds['tas'].sel(time=slice(pd.Timestamp('2020-01-20 00:00:00'), pd.Timestamp('2020-01-21 05:00:00'))).compute()
ds_plot = ds.isel(time=4).compute()

# %%
plot_field_for_times(da.isel(time=slice(1, None)), 'tas')
# %%
plot_all_fields(ds_plot)
