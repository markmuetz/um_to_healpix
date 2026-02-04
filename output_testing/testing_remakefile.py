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

def zooms_for_sim(sim):
    if sim == 'glm.n2560_RAL3p3.tuned':
        zooms = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    return zooms


class CreateLocalDataCache(Rule):
    atomic_output = False
    rule_matrix = {
        'sim': ['glm.n2560_RAL3p3.tuned'],
    }
    rule_inputs = {}
    @staticmethod
    def rule_outputs(sim):
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
        logger.debug(f'{inputs}, {outputs}, {sim}')
        config = load_config('../config/hk26_config.py')
        zooms = zooms_for_sim(sim)
        start_date = pd.Timestamp('2020-01-20 01:00:00')
        end_date = pd.Timestamp('2020-01-22 01:00:00')

        for zoom in zooms:
            output = outputs[f'pr_short.z{zoom}']
            if output.exists():
                logger.debug(f'{output} already created')
            else:
                ds = open_remote_dataset(config, sim, 'PT1H', zoom)
                logger.debug(f'cache precip: {sim}, {zoom}, {start_date} - {end_date}')
                ds.pr.sel(time=slice(start_date, end_date)).to_netcdf(output)

            output = outputs[f'all_fields.z{zoom}']
            if output.exists():
                logger.debug(f'{output} already created')
            else:
                ds = open_remote_dataset(config, sim, 'PT1H', zoom)
                # dicts and bools cannot be written to a .nc dataset.
                ds.attrs['bounds'] = str(ds.attrs['bounds'])
                ds.attrs['regional'] = str(ds.attrs['regional'])
                ds.sel(time=end_date).to_netcdf(output)

        output = outputs[f'pr_long.z0']
        if output.exists():
            logger.debug(f'{output} already created')
        else:
            ds = open_remote_dataset(config, sim, 'PT1H', 0)
            logger.debug(f'cache precip: {sim}, {zoom}, {start_date} -')
            ds.pr.to_netcdf(output)

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
    rule_matrix = {
        'sim': ['glm.n2560_RAL3p3.tuned'],
    }
    rule_inputs = CreateLocalDataCache.rule_outputs
    rule_outputs = {'dummy': 'outputs/{sim}/dummy.txt'}

    @staticmethod
    def rule_run(inputs, outputs, sim):
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

        PlotSanityChecks.plot_pr_short(sim, inputs, fig_outputs)
        PlotSanityChecks.plot_pr_med(inputs, fig_outputs)
        PlotSanityChecks.plot_pr_long(inputs, fig_outputs)
        PlotSanityChecks.plot_all_fields(sim, inputs, fig_outputs)

        outputs['dummy'].touch()

    @staticmethod
    def plot_pr_long(inputs, fig_outputs):
        ds = xr.open_dataset(inputs['pr_long.z0'])
        logger.info(np.isnan(ds.pr.values).sum())
        # Define dimensions
        width_px = 1920
        height_px = 1080
        dpi = 100

        # Create figure (width, height) in inches
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
        ds = xr.open_dataset(inputs['pr_med.z6'])
        umplt.plot_zonal_mean([ds.pr], bins=np.linspace(-90, 90, 181))
        plt.savefig(fig_outputs['pr_med.z6'])

    @staticmethod
    def plot_pr_short(sim, inputs, fig_outputs):
        zooms = zooms_for_sim(sim)
        prs = {}
        for zoom in zooms:
            if zoom < 6:
                continue
            logger.debug(f'cache precip {zoom}')
            ds = xr.open_dataset(inputs[f'pr_short.z{zoom}'])
            prs[zoom] = ds.pr
        umplt.plot_zonal_mean(prs.values(), bins=np.linspace(-90, 90, 181))
        plt.savefig(fig_outputs['pr_short.zonal_mean'])

    @staticmethod
    def plot_all_fields(sim, inputs, fig_outputs):
        zooms = zooms_for_sim(sim)
        for zoom in zooms:
            logger.debug(f'plot_all_fields {zoom}')
            ds = xr.open_dataset(inputs[f'all_fields.z{zoom}'])
            umplt.plot_all_fields(ds)
            plt.savefig(fig_outputs[f'all_fields.z{zoom}'])


