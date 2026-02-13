"""Remake workflow for repeatable output-testing data extraction and plotting.

This remake file defines two main rules:
1) ``CreateLocalDataCache``
    - Pulls selected remote HEALPix/Zarr data windows.
    - Stores compact local NetCDF caches for deterministic plotting tests.
2) ``PlotSanityChecks``
    - Consumes cached NetCDF files and generates diagnostic figures.

The intent is reproducible, scriptable sanity checking that is less expensive
than repeatedly reading full remote datasets during each plot run.
"""

from pathlib import Path

import numpy as np
from loguru import logger
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr

from remake import Remake, Rule
# from remake.util import tmp_to_actual_path

from um_to_healpix.util import load_config, open_remote_dataset
import um_to_healpix.plotting as umplt

rmk = Remake()

# Simulation set targeted by this remake workflow.
SIMS = ['glm.n2560_RAL3p3.tuned', 'glm.n1280_CoMA9']

def zooms_for_sim(sim):
    """Return valid HEALPix zoom list for a given simulation key."""
    if sim == 'glm.n2560_RAL3p3.tuned':
        zooms = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    elif sim == 'glm.n1280_CoMA9':
        zooms = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    return zooms


class CreateLocalDataCache(Rule):
    """Create local NetCDF caches used by downstream plotting checks.

    Output groups:
    - ``pr_short``: short precip window across all zooms.
    - ``pr_long``: long precip series at z0.
    - ``pr_med``: medium-duration precip window at z6.
    - ``all_fields``: one-time snapshot dataset per zoom.
    """

    # Disable atomic output to allow incremental file creation/checking per
    # generated output target.
    atomic_output = False

    # Expand rule once per simulation.
    rule_matrix = {
        'sim': SIMS,
    }

    # No explicit file inputs: this rule pulls remote data directly.
    rule_inputs = {}

    @staticmethod
    def rule_outputs(sim):
        """Declare output cache files for one simulation."""
        zooms = zooms_for_sim(sim)
        outputs = {f'pr_short.z{zoom}': Path(f'outputs/{sim}/pr_short.z{zoom}.nc')
                   for zoom in zooms}
        outputs.update({f'pr_long.z0': Path(f'outputs/{sim}/pr_long.z0.nc')})
        outputs.update({f'pr_med.z6': Path(f'outputs/{sim}/pr_med.z6.nc')})
        outputs.update({f'all_fields.z{zoom}': Path(f'outputs/{sim}/all_fields.z{zoom}.nc')
                        for zoom in zooms})
        return outputs

    @staticmethod
    def rule_run(inputs, outputs, sim):
        """Populate cache files if missing (idempotent file-by-file behavior)."""
        logger.debug(f'{inputs}, {outputs}, {sim}')
        config = load_config('../config/hk26_config.py')
        zooms = zooms_for_sim(sim)
        start_date = pd.Timestamp('2020-01-20 01:00:00')
        end_date = pd.Timestamp('2020-01-22 01:00:00')

        for zoom in zooms:
            # Short precip window cache per zoom.
            output = outputs[f'pr_short.z{zoom}']
            if output.exists():
                logger.debug(f'{output} already created')
            else:
                ds = open_remote_dataset(config, sim, 'PT1H', zoom)
                logger.debug(f'cache precip: {sim}, {zoom}, {start_date} - {end_date}')
                ds.pr.sel(time=slice(start_date, end_date)).to_netcdf(output)

            # Single-time all-fields cache per zoom for panel plotting.
            output = outputs[f'all_fields.z{zoom}']
            if output.exists():
                logger.debug(f'{output} already created')
            else:
                ds = open_remote_dataset(config, sim, 'PT1H', zoom)
                # dicts and bools cannot be written to a .nc dataset.
                ds.attrs['bounds'] = str(ds.attrs['bounds'])
                ds.attrs['regional'] = str(ds.attrs['regional'])
                ds.sel(time=end_date).to_netcdf(output)

        # Long precip cache at global-low zoom for long-run trend checks.
        output = outputs[f'pr_long.z0']
        if output.exists():
            logger.debug(f'{output} already created')
        else:
            ds = open_remote_dataset(config, sim, 'PT1H', 0)
            logger.debug(f'cache precip: {sim}, {zoom}, {start_date} -')
            ds.pr.to_netcdf(output)

        # Medium-duration precip cache at z6 for additional profile checks.
        output = outputs[f'pr_med.z6']
        if output.exists():
            logger.debug(f'{output} already created')
        else:
            ds = open_remote_dataset(config, sim, 'PT1H', 6)
            print(ds)
            logger.debug(f'cache precip: {sim}, {zoom}, {start_date} -')
            start_date = pd.Timestamp('2020-02-01 00:00:00')
            end_date = pd.Timestamp('2020-03-01 00:00:00')
            ds.pr.sel(time=slice(start_date, end_date)).to_netcdf(output)


class PlotSanityChecks(Rule):
    """Generate figure-based diagnostics from local cached NetCDF inputs."""

    # One plotting execution per simulation.
    rule_matrix = {
        'sim': SIMS,
    }

    # Consume all caches produced by the upstream rule.
    rule_inputs = CreateLocalDataCache.rule_outputs

    # Dummy output marker to represent completion of plotting bundle.
    rule_outputs = {'dummy': 'outputs/{sim}/dummy.txt'}

    @staticmethod
    def rule_run(inputs, outputs, sim):
        """Create all figure diagnostics and touch completion marker."""
        zooms = zooms_for_sim(sim)
        fig_outputs = {f'pr_short.zonal_mean': Path(f'outputs/figs/{sim}/pr_short.zonal_mean.png')}
        fig_outputs.update({f'pr_long.z0': Path(f'outputs/figs/{sim}/pr_long.z0.png')})
        fig_outputs.update({f'pr_med.z6': Path(f'outputs/figs/{sim}/pr_med.zonal_mean.z6.png')})
        fig_outputs.update({f'all_fields.z{zoom}': Path(f'outputs/figs/{sim}/all_fields.z{zoom}.png')
                            for zoom in zooms})

        outdirs = set(p.parent for p in fig_outputs.values())
        for outdir in outdirs:
            outdir.mkdir(parents=True, exist_ok=True)

        logger.debug(f'{inputs}, {outputs}, {sim}')

        # Produce each figure family using dedicated helper methods.
        PlotSanityChecks.plot_pr_short(sim, inputs, fig_outputs)
        PlotSanityChecks.plot_pr_med(inputs, fig_outputs)
        PlotSanityChecks.plot_pr_long(inputs, fig_outputs)
        PlotSanityChecks.plot_all_fields(sim, inputs, fig_outputs)

        # Mark rule completion.
        outputs['dummy'].touch()

    @staticmethod
    def plot_pr_long(inputs, fig_outputs):
        """Plot long precip timeseries at z0 with zoomed sub-window panel."""
        ds = xr.open_dataset(inputs['pr_long.z0'])
        logger.info(np.isnan(ds.pr.values).sum())
        # Define figure geometry explicitly for consistent readability.
        width_px = 1920
        height_px = 1080
        dpi = 100

        # Create two-panel figure: full history + selected zoom-in segment.
        timeslice = slice(4000, 5000)
        fig, axes = plt.subplots(2, 1, figsize=(width_px / dpi, 10), dpi=dpi, layout='constrained')
        hours = np.arange(len(ds.time))
        axes[0].plot(hours, ds.pr.mean(dim='cell'))
        axes[0].axvline(x=timeslice.start, c='k', ls='--')
        axes[0].axvline(x=timeslice.stop, c='k', ls='--')

        axes[1].plot(hours[timeslice], ds.pr.isel(time=timeslice).mean(dim='cell'))
        plt.xlabel('hour')
        for ax in axes:
            ax.set_ylabel(f'{ds.pr.long_name} ({ds.pr.attrs.get("units", "-")})')
        plt.savefig(fig_outputs['pr_long.z0'])

    @staticmethod
    def plot_pr_med(inputs, fig_outputs):
        """Plot zonal mean for medium-duration z6 precipitation cache."""
        ds = xr.open_dataset(inputs['pr_med.z6'])
        umplt.plot_zonal_mean([ds.pr], bins=np.linspace(-90, 90, 181))
        plt.savefig(fig_outputs['pr_med.z6'])

    @staticmethod
    def plot_pr_short(sim, inputs, fig_outputs):
        """Compare short-window zonal means across higher zoom levels."""
        zooms = zooms_for_sim(sim)
        prs = {}
        for zoom in zooms:
            # Restrict to higher zooms where short-range cross-zoom comparison is
            # most informative for this diagnostic.
            if zoom < 6:
                continue
            logger.debug(f'cache precip {zoom}')
            ds = xr.open_dataset(inputs[f'pr_short.z{zoom}'])
            prs[zoom] = ds.pr
        umplt.plot_zonal_mean(prs.values(), bins=np.linspace(-90, 90, 181))
        plt.savefig(fig_outputs['pr_short.zonal_mean'])

    @staticmethod
    def plot_all_fields(sim, inputs, fig_outputs):
        """Render all-field panel figures for each available zoom level."""
        zooms = zooms_for_sim(sim)
        for zoom in zooms:
            logger.debug(f'plot_all_fields {zoom}')
            ds = xr.open_dataset(inputs[f'all_fields.z{zoom}'])
            umplt.plot_all_fields(ds)
            plt.savefig(fig_outputs[f'all_fields.z{zoom}'])


