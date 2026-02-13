"""Interactive smoke-test notebook script for quick visual checks.

This file is intended for exploratory/manual testing in an IPython-enabled
environment. It focuses on opening one remote HEALPix dataset and generating a
small set of plots to verify basic field integrity and plotting behavior.

Notes:
- Uses remote object-store data via ``open_remote_dataset``.
- Designed to be run cell-by-cell (``# %%`` blocks) in VS Code/Jupyter style.
- This is not a formal automated test.
"""

# %% Autoreload setup (interactive quality-of-life).
from IPython import get_ipython

# Equivalent to:
#   %load_ext autoreload
#   %autoreload 2
# but executed safely via API to avoid syntax issues in plain Python mode.
if ipy := get_ipython():
    ipy.run_line_magic('load_ext', 'autoreload')
    ipy.run_line_magic('autoreload', '2')

# %% Imports.
# pandas: timestamp construction for selection windows.
# util helpers: config loading + remote dataset open.
# plotting: high-level convenience plotting functions for HEALPix fields.
import pandas as pd

from um_to_healpix.util import load_config, open_remote_dataset
from um_to_healpix.plotting import plot_field_for_times, plot_all_fields, plot_zonal_mean

# %% Settings.
# Load processing config from the HK26 config module.
config = load_config('config/hk26_config.py')

# Simulation key to inspect.
sim = 'glm.n2560_RAL3p3.tuned'

# Frequency controls which zarr store family is opened:
# - PT1H = hourly 2D diagnostics
# - PT3H = 3-hourly 3D diagnostics
freq = 'PT1H'  # 2D
# freq = 'PT3H'  # 3D

# HEALPix zoom level to inspect.
# At the time this script was written, z9 had representative data available.
zoom = 9  # Only zoom with any data in so far.

# Set True when running directly on JASMIN environment if required by helper
# logic (e.g., auth/path handling).
on_jasmin = False

# %% Open data and attach coords.
# Returns an xarray Dataset for the configured simulation/frequency/zoom.
ds = open_remote_dataset(config, sim, freq, zoom, on_jasmin)

# %% Load data window.
# Keep the time window tiny to minimize memory and IO for quick checks.
start_date = pd.Timestamp('2020-01-20 01:00:00')
end_date = pd.Timestamp('2020-01-20 01:00:00')

# Alternative examples (commented out) for quickly switching target fields.
# pr = ds['pr'].sel(time=slice(start_date, end_date)).compute()
# tas = ds['tas'].sel(time=slice(start_date, end_date)).compute()
clt = ds['clt'].sel(time=slice(start_date, end_date)).compute()


# %% Zonal-mean sanity plot.
# Useful for quickly spotting large-scale latitudinal structure anomalies.
plot_zonal_mean(clt)

# %% Global map plot(s) at selected time(s).
plot_field_for_times(clt, 'clt')
# plot_field_for_times(tas.isel(time=slice(1, None)), 'tas')

# %% Load a single-time full dataset snapshot.
# This is used to generate a panel of all fields available at one time step.
ds_plot = ds.isel(time=4).compute()

# %% Multi-field panel plot.
plot_all_fields(ds_plot)
