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

from um_to_healpix.util import load_config, open_remote_dataset
from um_to_healpix.plotting import plot_field_for_times, plot_all_fields, plot_zonal_mean

# %% Settings.
config = load_config('config/hk26_config.py')
sim = 'glm.n2560_RAL3p3.tuned'
freq = 'PT1H'  # 2D
# freq = 'PT3H'  # 3D
zoom = 9  # Only zoom with any data in so far.
on_jasmin = False

# %% Open data and attach coords.
ds = open_remote_dataset(config, sim, freq, zoom, on_jasmin)

# %% Load data.
start_date = pd.Timestamp('2020-01-20 01:00:00')
end_date = pd.Timestamp('2020-01-20 01:00:00')
# pr = ds['pr'].sel(time=slice(start_date, end_date)).compute()
# tas = ds['tas'].sel(time=slice(start_date, end_date)).compute()
clt = ds['clt'].sel(time=slice(start_date, end_date)).compute()


# %%
plot_zonal_mean(clt)

# %%
plot_field_for_times(clt, 'clt')
# plot_field_for_times(tas.isel(time=slice(1, None)), 'tas')

# %% Load all fields data.
ds_plot = ds.isel(time=4).compute()

# %%
plot_all_fields(ds_plot)
