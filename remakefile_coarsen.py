"""Remake3 remakefile for coarsening tasks.

Run from the repo root, in the pixi env, as regridding progresses:
    pixi run remake run remakefile_coarsen.py -E slurm

Two independent chains of one rule per zoom level:
    coarsen_2d_z9 → coarsen_2d_z8 → ... → coarsen_2d_z0
    coarsen_3d_z9 → coarsen_3d_z8 → ... → coarsen_3d_z0
Tasks within a rule run in parallel; each task coarsens NBATCH time chunks at its target zoom and is
named by the first time it covers (`start`), so e.g. -Q 'start < "2020-02-01"' selects by date.

A batch is only included once every .pp date its time range needs has a *successful* regrid task
(read from remakefile_regrid.py's remake DB via the remake Python API). So this can be run
incrementally as data is regridded — batches appear as their inputs complete.
N.B. there is no rerun propagation from regrid to coarsen: if a regrid task is rerun (e.g. after a
bug fix), force the affected coarsen batches with --force -Q.

As in remakefile_regrid.py, the processing config is loaded inside the rule and not tracked for reruns;
only the output location (deploy/output_vn) is. Per-task logs are also written to config.logdir.
"""
import math
from collections import defaultdict
from functools import cache
from pathlib import Path

import pandas as pd
from remake import Remake, rule

from um_to_healpix import checks
from um_to_healpix.util import load_config

CONFIG_PATH = Path('config/hk26_config.py')
REGRID_REMAKEFILE = 'remakefile_regrid.py'
config_module = load_config(CONFIG_PATH)
PROCESSING_CONFIG = config_module.processing_config
# Only the sims this pipeline is responsible for (config.remake_config_keys; default all).
CONFIG_KEYS = list(getattr(config_module, 'remake_config_keys', PROCESSING_CONFIG.keys()))
OUTPUT_LOCATION = {'deploy': config_module.deploy, 'output_vn': config_module.output_vn}

NBATCH = 10
DIMS = ['2d', '3d']

rmk = Remake(config={
    'slurm': {
        'account': config_module.slurm_config['account'],
        'partition': 'standard',
        # qos=standard rejects >1 CPU per job.
        'qos': 'high',
        'time': '10:00:00',
        'mem': '100G',
        # Keys are passed verbatim to #SBATCH --<key>=<value>.
        'cpus-per-task': 12,
        'export': 'ALL,OMP_NUM_THREADS=1',
        # Bad nodes, from the (untracked) config; empty -> no --exclude line.
        'exclude': config_module.slurm_config.get('exclude', ''),
    },
})


def _time_index(config, dim):
    time_idx = config.time2d if dim == '2d' else config.time3d
    # Drop the last time step (extends beyond input data).
    return time_idx[:-1]


@cache
def _regrid_done_dates():
    """config_key -> set of .pp dates (pd.Timestamp) whose regrid task has succeeded."""
    from remake import load_remake

    regrid_rmk = load_remake(REGRID_REMAKEFILE)
    query = 'rule == "regrid"'
    summary = regrid_rmk.status_summary(query=query, list_tasks=True)
    status = {row['key']: row['status'] for row in summary['tasks']}
    done = defaultdict(set)
    for task in regrid_rmk.iter_tasks(query):
        if status.get(task.key) == 'success':
            done[task.kwargs['config_key']].add(pd.Timestamp(task.kwargs['date']))
    return done


def _required_regrid_dates(times, first_date):
    """.pp dates (12-hourly files) needed to fill the target times.

    A time t is written by file floor(t, 12h) or, for hourly means (shifted from hh:30 to the following hour)
    and streams that run 01:00-12:00, by the previous file floor(t - 1h, 12h). Dates before the first input
    file have no data, so are not required.
    """
    required = set(times.floor('12h')) | set((times - pd.Timedelta(hours=1)).floor('12h'))
    return {d for d in required if d >= first_date}


def _batch_len(cfg, dim, zoom):
    return NBATCH * cfg['groups'][dim]['chunks'][zoom][0]


def _coarsen_matrix(dim, zoom):
    """(config_key, start) for every batch at this dim/zoom whose inputs are fully regridded."""
    done = _regrid_done_dates()
    time_idx = _time_index(config_module, dim)
    rows = []
    for config_key in CONFIG_KEYS:
        cfg = PROCESSING_CONFIG[config_key]
        if zoom >= cfg['max_zoom']:
            continue
        batch_len = _batch_len(cfg, dim, zoom)
        for batch_start in range(0, len(time_idx), batch_len):
            times = time_idx[batch_start:batch_start + batch_len]
            if _required_regrid_dates(times, cfg['first_date']) <= done[config_key]:
                rows.append({'config_key': config_key, 'start': f'{time_idx[batch_start]:%Y-%m-%dT%H}'})
    return rows


# Peak RSS measured over the p4k run (2026-09-13), rounded up with headroom: the 3d batches at z8/z7 read
# ~16x what a z9 batch does and OOMed at the old flat 32G (peak 42.8G). See docs/p4k_production_run_plan_2026-09-11.md.
COARSEN_MEM = {
    ('2d', 9): '8G',    # peak 2.6G
    ('3d', 9): '24G',   # peak 12.0G
    ('3d', 8): '64G',   # peak 42.7G
    ('3d', 7): '64G',   # peak 42.8G
}
COARSEN_MEM_DEFAULT = {'2d': '16G', '3d': '24G'}  # 2d z8-z0 peak <=6.0G; 3d z6-z0 peak <=10.7G


def _coarsen_mem(dim, zoom):
    return COARSEN_MEM.get((dim, zoom), COARSEN_MEM_DEFAULT[dim])


def _make_coarsen_rule(dim, zoom, upstream):
    def matrix():
        return _coarsen_matrix(dim, zoom)

    @rule(
        name=f'coarsen_{dim}_z{zoom}',
        matrix=matrix,
        depends_on=[upstream] if upstream is not None else [],
        uses={
            'dim': dim,
            'zoom': zoom,
            'NBATCH': NBATCH,
            'OUTPUT_LOCATION': OUTPUT_LOCATION,
            '_time_index': _time_index,
            '_batch_len': _batch_len,
        },
        # These tasks are I/O bound on S3: they used 0.11 (2d) / 0.30 (3d) cores of the 12 previously
        # requested, and aggregate throughput was S3-limited (~215 batches/h) regardless of concurrency.
        config={'slurm': {'mem': _coarsen_mem(dim, zoom), 'cpus-per-task': 2, 'array_throttle': 60}},
    )
    def coarsen(config_key, start):
        import pandas as pd
        from um_to_healpix.um_process_tasks import UMProcessTasks
        from um_to_healpix.util import load_config, task_log

        config = load_config(CONFIG_PATH)
        cfg = config.processing_config[config_key]
        time_idx = _time_index(config, dim)
        timechunk = cfg['groups'][dim]['chunks'][zoom][0]
        njobs = int(math.ceil(len(time_idx) / timechunk))

        batch_start = time_idx.get_loc(pd.Timestamp(start))
        assert batch_start % _batch_len(cfg, dim, zoom) == 0, f'{start} is not the start of a batch'
        start_job = batch_start // timechunk
        end_job = min(start_job + NBATCH, njobs)
        tgt_times = [
            {'start_idx': i * timechunk, 'end_idx': (i + 1) * timechunk}
            for i in range(start_job, end_job)
        ]

        task = {
            'task_type': 'coarsen',
            'config_path': str(CONFIG_PATH),
            'config_key': config_key,
            'tgt_zoom': zoom,
            'dim': dim,
            'tgt_times': tgt_times,
        }
        with task_log(config.logdir / config_key / 'coarsen' / dim / f'z{zoom}' / f'{start}.log'):
            proc = UMProcessTasks(cfg, config.shared_metadata)
            proc.coarsen_healpix_region(task)

    return coarsen


# Build one zoom chain per dim: coarsen_<dim>_z{MAX_ZOOM-1} has no upstream, then each zoom depends on the next
# highest. Sims with a lower max_zoom (N1280: 9) have no tasks in the higher rules.
MAX_ZOOM = max(PROCESSING_CONFIG[k]['max_zoom'] for k in CONFIG_KEYS)

coarsen_rules = []
final_rules = {}  # dim -> the z0 rule, i.e. the end of that dim's chain
for dim in DIMS:
    prev_rule = None
    for zoom in range(MAX_ZOOM - 1, -1, -1):
        prev_rule = _make_coarsen_rule(dim, zoom, prev_rule)
        coarsen_rules.append(prev_rule)
    final_rules[dim] = prev_rule

rmk.add_rules(coarsen_rules)

# Checks
# ======
# One task per written store (freq x zoom), after *both* chains have finished, writing a JSON report.
# A failing check raises, so the report is only written for a store that passed - and, being a declared output,
# `remake info`/`set-state --check-outputs` can see which stores are verified.
FREQ_DIM = {'PT1H': '2d', 'PT3H': '3d'}
# Time steps to sample for the value/consistency checks (fractions through the series).
CHECK_TIME_FRACTIONS = [0.1, 0.5, 0.9]


def _report_path(config_key, freq, zoom):
    return config_module.logdir / 'checks' / config_key / f'{freq}_z{zoom}.json'


def check_outputs(config_key, freq, zoom):
    return {'report': _report_path(config_key, freq, zoom)}


@rule(
    name='check_store',
    matrix={'config_key': CONFIG_KEYS, 'freq': list(FREQ_DIM), 'zoom': list(range(MAX_ZOOM + 1))},
    outputs=check_outputs,
    depends_on=[final_rules[d] for d in DIMS],
    # The check functions and their thresholds are tracked, so tightening a check re-runs the checks
    # (the rule body imports um_to_healpix.checks at run time, which remake cannot see by itself).
    uses={'FREQ_DIM': FREQ_DIM, 'CHECK_TIME_FRACTIONS': CHECK_TIME_FRACTIONS,
          '_time_index': _time_index, 'OUTPUT_LOCATION': OUTPUT_LOCATION,
          'check_structure': checks.check_structure, 'check_coverage': checks.check_coverage,
          'check_nans_and_ranges': checks.check_nans_and_ranges,
          'check_level_consistency': checks.check_level_consistency,
          'written_time_steps': checks.written_time_steps,
          'check_thresholds': {'ranges': checks.RANGE_CHECKS, 'range_atol': checks.RANGE_CHECK_ATOL,
                               'nan_global': checks.MAX_NAN_FRACTION_GLOBAL,
                               'nan_by_var': checks.MAX_NAN_FRACTION_BY_VAR,
                               'consistency_rtol': checks.CONSISTENCY_RTOL,
                               'consistency_atol_frac': checks.CONSISTENCY_ATOL_FRAC,
                               'max_consistency_zoom': checks.MAX_CONSISTENCY_ZOOM}},
    config={'slurm': {'mem': '32G', 'cpus-per-task': 2, 'array_throttle': 22}},
)
def check_store(outputs, config_key, freq, zoom):
    import json

    from um_to_healpix import checks
    from um_to_healpix.um_process_tasks import get_jasmin_s3
    from um_to_healpix.util import load_config, task_log

    config = load_config(CONFIG_PATH)
    cfg = config.processing_config[config_key]
    dim = FREQ_DIM[freq]
    url = cfg['zarr_store_url_tpl'].format(freq=freq, zoom=zoom)

    with task_log(config.logdir / config_key / 'checks' / f'{freq}_z{zoom}.log'):
        fs = get_jasmin_s3()
        ds = checks.open_store(url, fs)
        time_idx = [int(f * (ds.sizes['time'] - 1)) for f in CHECK_TIME_FRACTIONS]
        # The store holds every time step of the config index; the regridded data starts one step in for
        # time-means, which check_coverage reports per variable rather than failing on.
        expected_time = config.time2d if dim == '2d' else config.time3d
        # A store holds every group that targets it: PT1H has 2d *and* 2d_depth (mrsol), PT3H has 3d and 3d_ml.
        expected_vars = [short
                         for group in cfg['groups'].values() if group['zarr_store'] == freq
                         for short, _ in group['name_map']]

        report = {'url': url, 'freq': freq, 'zoom': zoom, 'config_key': config_key,
                  'n_time': int(ds.sizes['time']), 'time_idx_checked': time_idx, 'failures': {}}
        report['failures']['structure'] = checks.check_structure(
            ds, expected_time, 12 * 4 ** zoom, expected_vars=expected_vars)
        report['failures']['coverage'] = checks.check_coverage(fs, url, ds)
        nan_range = []
        for i in time_idx:
            nan_range += checks.check_nans_and_ranges(ds, i)
        report['failures']['nans_and_ranges'] = nan_range

        # A coarsened field must be the mean of its 4 children. Only worth reading whole fields at low zoom.
        consistency = []
        if zoom < cfg['max_zoom'] and zoom <= checks.MAX_CONSISTENCY_ZOOM:
            ds_fine = checks.open_store(cfg['zarr_store_url_tpl'].format(freq=freq, zoom=zoom + 1), fs)
            for i in time_idx:
                consistency += checks.check_level_consistency(ds, ds_fine, i)
        report['failures']['consistency'] = consistency

        report['passed'] = not any(report['failures'].values())
        path = Path(outputs['report'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str))

        if not report['passed']:
            summary = {k: len(v) for k, v in report['failures'].items() if v}
            raise Exception(f'{freq} z{zoom} failed checks {summary} - see {path}')


rmk.add_rules([check_store])

# Sanity plots
# ============
# Only produced for stores that passed their checks (depends_on=check_store), using the same plotting helpers as
# output_tests/test_plots.py. Figures land next to the reports so a run can be eyeballed quickly.
PLOT_VAR = {'PT1H': 'pr', 'PT3H': 'ta'}      # zonal mean / timeseries variable per store
PLOT_LEVEL = {'PT3H': 500}                    # pressure level (hPa) for 3d variables
# The default 0.1-degree zonal bins only make sense where there are enough cells: at z3 just 31 of 1800 bins are
# populated, so the line is mostly NaN. Use 1-degree bins and only zooms with >= ~10k cells for the zonal mean.
ZONAL_ZOOMS = [5, 7]
ZONAL_BIN_DEG = 1.0
TIMESERIES_ZOOMS = [0, 3, 5, 7]               # domain means are meaningful at every zoom
SNAPSHOT_ZOOMS = [0, 5, 8]                    # zooms for the all-fields map plots
PLOT_NTIME = 240                              # time steps averaged (10 days hourly, 30 days 3-hourly)


def _fig_path(config_key, freq, name):
    return config_module.logdir / 'figures' / config_key / f'{freq}_{name}.png'


def _open_for_plot(config, cfg, freq, zoom, fs):
    """Open a store with lat/lon coords attached, as the plotting helpers expect.

    The stores name the dimension `healpix_index`; easygems attaches lat/lon against `cell` (as the published
    catalogue datasets use), so rename first or the plotting helpers see no `lat` coord.
    """
    import easygems.healpix as egh
    from um_to_healpix import checks
    ds = checks.open_store(cfg['zarr_store_url_tpl'].format(freq=freq, zoom=zoom), fs)
    if 'healpix_index' in ds.dims:
        ds = ds.rename({'healpix_index': 'cell'})
    return egh.attach_coords(ds)


def zonal_mean_outputs(config_key, freq):
    return {'zonal_mean': _fig_path(config_key, freq, f'{PLOT_VAR[freq]}_zonal_mean'),
            'timeseries': _fig_path(config_key, freq, f'{PLOT_VAR[freq]}_timeseries')}


@rule(
    name='plot_zonal_mean',
    matrix={'config_key': CONFIG_KEYS, 'freq': list(FREQ_DIM)},
    outputs=zonal_mean_outputs,
    depends_on=[check_store],
    uses={'PLOT_VAR': PLOT_VAR, 'PLOT_LEVEL': PLOT_LEVEL, 'ZONAL_ZOOMS': ZONAL_ZOOMS,
          'ZONAL_BIN_DEG': ZONAL_BIN_DEG, 'TIMESERIES_ZOOMS': TIMESERIES_ZOOMS, 'PLOT_NTIME': PLOT_NTIME,
          '_open_for_plot': _open_for_plot, 'OUTPUT_LOCATION': OUTPUT_LOCATION},
    config={'slurm': {'mem': '32G', 'cpus-per-task': 2}},
)
def plot_zonal_mean(outputs, config_key, freq):
    """Zonal mean and domain-mean timeseries of one variable, overlaid across zooms.

    The zooms should lie on top of each other: coarsening conserves the field, so a disagreement between zoom
    levels is the signal this plot exists to show.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import um_to_healpix.plotting as umplt
    from um_to_healpix.um_process_tasks import get_jasmin_s3
    from um_to_healpix.util import load_config, task_log

    config = load_config(CONFIG_PATH)
    cfg = config.processing_config[config_key]
    var = PLOT_VAR[freq]

    with task_log(config.logdir / config_key / 'plots' / f'{freq}_zonal_mean.log'):
        import numpy as np

        fs = get_jasmin_s3()
        das = {}
        for zoom in sorted(set(ZONAL_ZOOMS) | set(TIMESERIES_ZOOMS)):
            if zoom > cfg['max_zoom']:
                continue
            ds = _open_for_plot(config, cfg, freq, zoom, fs)
            da = ds[var].isel(time=slice(0, PLOT_NTIME))
            if 'pressure' in da.dims:
                da = da.sel(pressure=PLOT_LEVEL[freq], method='nearest')
            das[zoom] = da.compute()

        bins = np.linspace(-90, 90, int(180 / ZONAL_BIN_DEG) + 1)
        zonal = [da for zoom, da in das.items() if zoom in ZONAL_ZOOMS]
        timeseries = {zoom: da for zoom, da in das.items() if zoom in TIMESERIES_ZOOMS}
        for key, plot in [('zonal_mean', lambda: umplt.plot_zonal_mean(zonal, bins=bins)),
                          ('timeseries', lambda: umplt.plot_timeseries(timeseries))]:
            path = Path(outputs[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(layout='constrained')
            plot()
            plt.title(f'{config_key} {freq} {var} ({key.replace("_", " ")}, first {PLOT_NTIME} steps)')
            plt.savefig(path, dpi=110)
            plt.close('all')


def snapshot_outputs(config_key, freq, zoom):
    return {'all_fields': _fig_path(config_key, freq, f'all_fields_z{zoom}')}


@rule(
    name='plot_all_fields',
    matrix={'config_key': CONFIG_KEYS, 'freq': list(FREQ_DIM), 'zoom': SNAPSHOT_ZOOMS},
    outputs=snapshot_outputs,
    depends_on=[check_store],
    uses={'PLOT_LEVEL': PLOT_LEVEL, '_open_for_plot': _open_for_plot, 'OUTPUT_LOCATION': OUTPUT_LOCATION},
    config={'slurm': {'mem': '32G', 'cpus-per-task': 2}},
)
def plot_all_fields(outputs, config_key, freq, zoom):
    """Every variable in the store on a world map, at one time step."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import um_to_healpix.plotting as umplt
    from um_to_healpix.um_process_tasks import get_jasmin_s3
    from um_to_healpix.util import load_config, task_log

    config = load_config(CONFIG_PATH)
    cfg = config.processing_config[config_key]
    if zoom > cfg['max_zoom']:
        return

    with task_log(config.logdir / config_key / 'plots' / f'{freq}_all_fields_z{zoom}.log'):
        ds = _open_for_plot(config, cfg, freq, zoom, get_jasmin_s3())
        snapshot = ds.isel(time=ds.sizes['time'] // 2)
        if 'pressure' in snapshot.dims:
            snapshot = snapshot.sel(pressure=PLOT_LEVEL[freq], method='nearest')
        umplt.plot_all_fields(snapshot.compute())
        path = Path(outputs['all_fields'])
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=110)
        plt.close('all')


rmk.add_rules([plot_zonal_mean, plot_all_fields])
