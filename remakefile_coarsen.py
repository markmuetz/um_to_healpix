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

from um_to_healpix.util import load_config

CONFIG_PATH = Path('config/hk26_config.py')
REGRID_REMAKEFILE = 'remakefile_regrid.py'
config_module = load_config(CONFIG_PATH)
PROCESSING_CONFIG = config_module.processing_config
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
    for config_key, cfg in PROCESSING_CONFIG.items():
        if zoom >= cfg['max_zoom']:
            continue
        batch_len = _batch_len(cfg, dim, zoom)
        for batch_start in range(0, len(time_idx), batch_len):
            times = time_idx[batch_start:batch_start + batch_len]
            if _required_regrid_dates(times, cfg['first_date']) <= done[config_key]:
                rows.append({'config_key': config_key, 'start': f'{time_idx[batch_start]:%Y-%m-%dT%H}'})
    return rows


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
        config={'slurm': {'mem': '100G'}},
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
MAX_ZOOM = max(cfg['max_zoom'] for cfg in PROCESSING_CONFIG.values())

coarsen_rules = []
for dim in DIMS:
    prev_rule = None
    for zoom in range(MAX_ZOOM - 1, -1, -1):
        prev_rule = _make_coarsen_rule(dim, zoom, prev_rule)
        coarsen_rules.append(prev_rule)

rmk.add_rules(coarsen_rules)
