"""Remake3 remakefile for coarsening tasks.

Run after all regridding is complete (from the repo root, in the pixi env):
    pixi run remake run remakefile_coarsen.py -E slurm

Generates one rule per zoom level (coarsen_z9 → coarsen_z8 → ... → coarsen_z0),
with tasks within each zoom level running in parallel.

As in remakefile_regrid.py, the processing config is loaded inside the rule and not tracked for reruns;
only the output location (deploy/output_vn) is.
"""
import math
from pathlib import Path

from remake import Remake, rule

from um_to_healpix.util import load_config

CONFIG_PATH = Path('config/hk26_config.py')
config_module = load_config(CONFIG_PATH)
PROCESSING_CONFIG = config_module.processing_config
OUTPUT_LOCATION = {'deploy': config_module.deploy, 'output_vn': config_module.output_vn}

NBATCH = 10

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
    },
})


def _time_index(config, dim):
    time_idx = config.time2d if dim == '2d' else config.time3d
    # Drop the last time step (extends beyond input data).
    return time_idx[:-1]


def _build_coarsen_matrix(zoom):
    """Return (config_key, dim, batch_id) tuples for this zoom level.

    Each batch_id groups NBATCH time chunks into one task (matching the
    existing batching logic in um_slurm_control.coarsen).
    """
    rows = []
    for config_key, cfg in PROCESSING_CONFIG.items():
        if zoom >= cfg['max_zoom']:
            continue
        for dim in ['2d', '3d']:
            timechunk = cfg['groups'][dim]['chunks'][zoom][0]
            njobs = int(math.ceil(len(_time_index(config_module, dim)) / timechunk))
            for batch_start in range(0, njobs, NBATCH):
                rows.append({
                    'config_key': config_key,
                    'dim': dim,
                    'batch_id': batch_start // NBATCH,
                })
    return rows


def _make_coarsen_rule(zoom, upstream):
    matrix = _build_coarsen_matrix(zoom)

    @rule(
        matrix=matrix,
        depends_on=[upstream] if upstream is not None else [],
        uses={
            'zoom': zoom,
            'NBATCH': NBATCH,
            'OUTPUT_LOCATION': OUTPUT_LOCATION,
            '_time_index': _time_index,
        },
        config={'slurm': {'mem': '100G'}},
    )
    def coarsen(config_key, dim, batch_id):
        from um_to_healpix.um_process_tasks import UMProcessTasks
        from um_to_healpix.util import load_config

        config = load_config(CONFIG_PATH)
        cfg = config.processing_config[config_key]
        time_idx = _time_index(config, dim)

        timechunk = cfg['groups'][dim]['chunks'][zoom][0]
        njobs = int(math.ceil(len(time_idx) / timechunk))

        start_job = batch_id * NBATCH
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
        proc = UMProcessTasks(cfg, config.shared_metadata)
        proc.coarsen_healpix_region(task)

    coarsen.fn.__name__ = f'coarsen_z{zoom}'
    coarsen.fn.__qualname__ = f'coarsen_z{zoom}'
    return coarsen


# Build the zoom chain: coarsen_z{MAX_ZOOM-1} has no upstream, coarsen_z8 depends on coarsen_z9, etc.
MAX_ZOOM = max(cfg['max_zoom'] for cfg in PROCESSING_CONFIG.values())

coarsen_rules = []
prev_rule = None
for zoom in range(MAX_ZOOM - 1, -1, -1):
    r = _make_coarsen_rule(zoom, prev_rule)
    coarsen_rules.append(r)
    prev_rule = r

rmk.add_rules(coarsen_rules)
