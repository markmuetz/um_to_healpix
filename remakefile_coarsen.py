"""Remake3 remakefile for coarsening tasks.

Run after all regridding is complete:
    remake run remakefile_coarsen.py -E slurm

Generates one rule per zoom level (coarsen_z9 → coarsen_z8 → ... → coarsen_z0),
with tasks within each zoom level running in parallel.
"""
import math
from itertools import batched
from pathlib import Path

from remake import Remake, rule

from um_to_healpix.util import load_config

CONFIG_PATH = Path('config/hk26_config.py')
config_module = load_config(CONFIG_PATH)
PROCESSING_CONFIG = config_module.processing_config

NBATCH = 10

rmk = Remake(config={
    'slurm': {
        'account': config_module.slurm_config['account'],
        'partition': 'standard',
        'qos': 'high',
        'time': '10:00:00',
        'mem': '100G',
        'cpus_per_task': 12,
    },
})


def _get_max_zoom(config_key):
    return PROCESSING_CONFIG[config_key]['max_zoom']


def _build_coarsen_matrix(zoom):
    """Return (config_key, dim, batch_id) tuples for this zoom level.

    Each batch_id groups NBATCH time chunks into one task (matching the
    existing batching logic in um_slurm_control.coarsen).
    """
    rows = []
    for config_key, cfg in PROCESSING_CONFIG.items():
        max_zoom = cfg['max_zoom']
        if zoom >= max_zoom:
            continue
        for dim in ['2d', '3d']:
            if dim == '2d':
                time_idx = config_module.time2d
            else:
                time_idx = config_module.time3d
            # Drop the last time step (extends beyond input data).
            time_idx = time_idx[:-1]

            chunks = cfg['groups'][dim]['chunks']
            timechunk = chunks[zoom][0]
            njobs = int(math.ceil(len(time_idx) / timechunk))

            for batch_start in range(0, njobs, NBATCH):
                batch_id = batch_start // NBATCH
                rows.append({
                    'config_key': config_key,
                    'dim': dim,
                    'batch_id': batch_id,
                })
    return rows


def _make_coarsen_rule(zoom, upstream):
    matrix = _build_coarsen_matrix(zoom)

    @rule(
        matrix=matrix,
        depends_on=[upstream] if upstream is not None else [],
        uses={
            'PROCESSING_CONFIG': PROCESSING_CONFIG,
            'config_module': config_module,
            'zoom': zoom,
            'NBATCH': NBATCH,
        },
        config={
            'slurm': {
                'mem': '100G',
                'cpus_per_task': 12,
            },
        },
    )
    def coarsen(config_key, dim, batch_id):
        from um_to_healpix.um_process_tasks import UMProcessTasks

        cfg = PROCESSING_CONFIG[config_key]

        if dim == '2d':
            time_idx = config_module.time2d
        else:
            time_idx = config_module.time3d
        time_idx = time_idx[:-1]

        chunks = cfg['groups'][dim]['chunks']
        timechunk = chunks[zoom][0]
        njobs = int(math.ceil(len(time_idx) / timechunk))

        start_job = batch_id * NBATCH
        end_job = min(start_job + NBATCH, njobs)

        tgt_times = []
        for i in range(start_job, end_job):
            tgt_times.append({
                'start_idx': i * timechunk,
                'end_idx': (i + 1) * timechunk,
            })

        task = {
            'task_type': 'coarsen',
            'config_path': str(CONFIG_PATH),
            'config_key': config_key,
            'tgt_zoom': zoom,
            'dim': dim,
            'tgt_times': tgt_times,
        }
        proc = UMProcessTasks(cfg, config_module.shared_metadata)
        proc.coarsen_healpix_region(task)

    coarsen.fn.__name__ = f'coarsen_z{zoom}'
    coarsen.fn.__qualname__ = f'coarsen_z{zoom}'
    return coarsen


# Build the zoom chain: coarsen_z9 depends on coarsen_z10 (which doesn't exist,
# so z9 has no upstream), coarsen_z8 depends on coarsen_z9, etc.
MAX_ZOOM = max(cfg['max_zoom'] for cfg in PROCESSING_CONFIG.values())

coarsen_rules = []
prev_rule = None
for zoom in range(MAX_ZOOM - 1, -1, -1):
    r = _make_coarsen_rule(zoom, prev_rule)
    coarsen_rules.append(r)
    prev_rule = r

rmk.add_rules(coarsen_rules)
