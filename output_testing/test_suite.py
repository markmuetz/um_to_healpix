"""Manual plotting test suite for HEALPix output sanity checks.

This script is aimed at interactive/manual verification rather than unit tests.
It opens remote datasets across zoom levels, caches computed selections locally,
and generates a suite of diagnostic plots to inspect:

- precipitation consistency across zooms,
- all-field plotting robustness,
- model-level interpolated 3D variable behavior.
"""

# %% Autoreload setup.
# Skip if not running interactively (no active IPython shell).
from IPython import get_ipython

if ipy := get_ipython():
    ipy.run_line_magic('load_ext', 'autoreload')
    ipy.run_line_magic('autoreload', '2')

# %% Imports and cache utilities.
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import pandas as pd
from matplotlib import pyplot as plt

from um_to_healpix.util import load_config, open_remote_dataset
import um_to_healpix.plotting as umplt


@dataclass
class TestOpts:
    """Per-simulation options used by this script."""
    max_zoom: int

# Simulation-specific testing settings.
options = {
    'glm.n2560_RAL3p3.tuned': TestOpts(max_zoom=10),
}

class DataCache:
    """Lazy compute cache for xarray selections used repeatedly in tests.

    Workflow:
    1) Open all requested (freq, zoom) datasets once.
    2) For each query (field/sel/isel), compute lazily on first request.
    3) Store computed result in ``_cache`` keyed by query args.
    """
    freqs = ['PT1H', 'PT3H']
    def __init__(self, config, sim):
        # Keep config/simulation context for repeated lookups.
        self.config = config
        self.sim = sim
        self.opt = options[sim]

        # Iterate highest -> lowest zoom for plotting consistency.
        self.zooms = list(range(self.opt.max_zoom, -1, -1))

        # Open all required dataset handles eagerly (compute remains lazy).
        self.datasets = {}
        for freq in self.freqs:
            for zoom in self.zooms:
                logger.debug(f'{sim}/{freq}/{zoom}')
                ds = open_remote_dataset(config, sim, freq, zoom)
                self.datasets[(freq, zoom)] = ds

        # Query-result cache of computed xarray objects.
        self._cache = {}

    def __call__(self, **kwargs):
        """Query helper with optional sel/isel/field projection + compute cache."""
        key = str(kwargs)
        if key not in self._cache:
            freq = kwargs['freq']
            zoom = kwargs['zoom']
            ds = self.datasets[(freq, zoom)]

            # Optional selection/indexing/filtering pipeline.
            if 'sel' in kwargs:
                ds = ds.sel(**kwargs['sel'])
            if 'isel' in kwargs:
                ds = ds.isel(**kwargs['isel'])
            if 'field' in kwargs:
                ds = ds[kwargs['field']]

            # Materialize and store to avoid repeated remote reads/computes.
            self._cache[key] = ds.compute()
        return self._cache[key]



# %% Test suite implementation.
class TestSuite:
    """Collection of high-level manual sanity checks producing diagnostic figs."""
    def __init__(self, config, cache):
        self.config = config
        self.cache = cache

    def run(self):
        """Run all configured plotting checks."""
        self.plot_precip()
        self.plot_all_fields()
        self.plot_3d_interped_field()

    def savefig(self, path):
        """Save current figure, creating parent directories if needed."""
        path = Path(path)
        logger.debug(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)

    def plot_precip(self):
        """Precipitation checks: zonal mean + global mean timeseries diagnostics."""
        start_date = pd.Timestamp('2020-01-20 01:00:00')
        # This pretty much maxes out the RAM on my laptop (16+8swap).
        # end_date = pd.Timestamp('2020-01-23 01:00:00')
        end_date = pd.Timestamp('2020-01-22 01:00:00')
        prs = {}

        # Compare precip field across all zoom levels over same time range.
        for zoom in self.cache.zooms:
            logger.debug(f'cache precip {zoom}')
            prs[zoom] = self.cache(freq='PT1H', zoom=zoom, field='pr', sel=dict(time=slice(start_date, end_date)))
        umplt.plot_zonal_mean(prs.values())
        plt.title(f'{self.cache.sim}: {start_date} - {end_date}')
        self.savefig(f'output_testing/figures/{self.config.output_vn}/{sim}/PT1H/pr_zonal_mean.png')

        umplt.plot_timeseries(prs)

        plt.title(f'Global precip, {self.cache.sim}: {start_date} - {end_date}')
        self.savefig(f'output_testing/figures/{self.config.output_vn}/{sim}/PT1H/pr_timeseries.png')

        # Longer-window check at one moderate zoom to spot temporal artifacts.
        end_date = pd.Timestamp('2020-02-21 01:00:00')
        zoom = 4
        pr = self.cache(freq='PT1H', zoom=zoom, field='pr', sel=dict(time=slice(start_date, end_date)))
        umplt.plot_timeseries({zoom: pr})
        plt.title(f'Global precip, {self.cache.sim}: {start_date} - {end_date}')
        self.savefig(f'output_testing/figures/{self.config.output_vn}/{sim}/PT1H/pr_timeseries.long.z4.png')

    def plot_all_fields(self):
        """Render all available PT1H fields for one timestamp at each zoom."""
        date = pd.Timestamp('2020-02-20 01:00:00')
        for zoom in self.cache.zooms:
            logger.debug(f'plot_all_fields {zoom}')
            # For the lower zooms, selecting one time is the worse cast scenario as it needs to load
            # all the time data anyway due to chunking.
            ds = self.cache(freq='PT1H', zoom=zoom, sel=dict(time=date))
            umplt.plot_all_fields(ds)
            self.savefig(f'output_testing/figures/{self.config.output_vn}/{sim}/PT1H/'
                         f'all_fields.{zoom}.{str(date).replace(" ", "_")}.png')

    def plot_3d_interped_field(self):
        """Quick check of model-level -> pressure-interpolated 3D variable profile."""
        # name_map_3d_ml is on model levels, i.e. interped fields.
        assert ('clw', 'mass_fraction_of_cloud_liquid_water_in_air') in self.config.name_map_3d_ml

        zoom = 5
        time = '2020-01-25 00:00'
        clw = self.cache(freq='PT3H', zoom=zoom, field='clw', sel=dict(time=time))

        # Plot vertical mean profile over all cells.
        plt.plot(clw.mean(dim='cell'), clw.pressure)
        plt.ylim((1000, 0))
        plt.xlabel(f'{clw.long_name} ({clw.attrs.get("units", "-")})')
        plt.title(f'{time} - z{zoom}')
        self.savefig(f'output_testing/figures/{self.config.output_vn}/{sim}/PT1H/'
                     f'interped_field.clw.z{zoom}.{str(time).replace(" ", "_")}.png')


# %% Setup and run.
# Load configuration + initialize dataset cache for target simulation.
config = load_config('config/hk26_config.py')
sim = 'glm.n2560_RAL3p3.tuned'
cache = DataCache(config, sim)

# %% Execute suite.
test_suite = TestSuite(config, cache)
test_suite.run()

# %%
# Ad-hoc follow-up diagnostic:
# These fields can appear unusual due to colormap auto-selection heuristics in
# umplt.get_plot_args (diverging vs sequential), not necessarily data issues.
hflsd9 = cache(freq='PT1H', zoom=9, field='hflsd', sel=dict(time='2020-02-20 01:00:00'))
hflsd8 = cache(freq='PT1H', zoom=8, field='hflsd', sel=dict(time='2020-02-20 01:00:00'))

# %%
# Inspect current cache keys to understand memory usage/state.
print(cache._cache.keys())

# %%
# Optional cleanup to release memory before running additional ad-hoc checks.
cache._cache.clear()