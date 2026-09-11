"""Remake3 remakefile for create_empty_zarr_stores and regrid tasks.

Run when .pp files are available (from the repo root, in the pixi env):
    pixi run remake run remakefile_regrid.py -E slurm

Can be rerun as more .pp files land — remake3 skips completed tasks.

The processing config is deliberately *not* tracked for reruns (its repr embeds memory addresses,
and config edits should not silently rerun everything) - it is loaded inside each rule.
Only the output location (deploy/output_vn) is tracked. Rerun after config edits with --force -Q.
"""
from functools import cache
from pathlib import Path

from remake import Remake, rule

from um_to_healpix.um_process_tasks import UMProcessTasks
from um_to_healpix.um_slurm_control import DEFAULT_PP_GLOB, find_dyamond3_pp_dates_to_paths
from um_to_healpix.util import load_config

CONFIG_PATH = Path('config/hk26_config.py')
config_module = load_config(CONFIG_PATH)
PROCESSING_CONFIG = config_module.processing_config
CONFIG_KEYS = list(PROCESSING_CONFIG.keys())
OUTPUT_LOCATION = {'deploy': config_module.deploy, 'output_vn': config_module.output_vn}

rmk = Remake(config={
    'slurm': {
        'account': config_module.slurm_config['account'],
        'partition': 'standard',
        'qos': 'standard',
        'time': '10:00:00',
        'mem': '100G',
        # Keys are passed verbatim to #SBATCH --<key>=<value>.
        'export': 'ALL,OMP_NUM_THREADS=1',
    },
})


@cache
def _dates_to_paths(config_key):
    cfg = PROCESSING_CONFIG[config_key]
    return find_dyamond3_pp_dates_to_paths(cfg['basedir'], cfg.get('pp_glob', DEFAULT_PP_GLOB))


def _inpaths_dict(paths):
    return {'inpath_' + str(i): str(p) for i, p in enumerate(paths)}


def create_inputs(config_key):
    """Return first-date .pp paths for this config_key."""
    dates_to_paths = _dates_to_paths(config_key)
    first_date = PROCESSING_CONFIG[config_key]['first_date']
    if first_date not in dates_to_paths:
        raise ValueError(f'First date {first_date} not found for {config_key}')
    return _inpaths_dict(dates_to_paths[first_date])


@rule(
    inputs=create_inputs,
    matrix={'config_key': CONFIG_KEYS},
    uses={'UMProcessTasks': UMProcessTasks, 'OUTPUT_LOCATION': OUTPUT_LOCATION},
    # Generates N2560 z10 weights (30-40G each) if they do not exist.
    config={'slurm': {'mem': '100G', 'time': '24:00:00'}},
)
def create_stores(inputs, config_key):
    from um_to_healpix.util import load_config
    config = load_config(CONFIG_PATH)
    task = {
        'task_type': 'create_empty_zarr_stores',
        'config_path': str(CONFIG_PATH),
        'config_key': config_key,
        'inpaths': list(inputs.values()),
    }
    proc = UMProcessTasks(config.processing_config[config_key], config.shared_metadata)
    proc.create_empty_zarr_stores(task)


def regrid_matrix():
    """Scan input directories, return (config_key, date) pairs for available .pp files."""
    rows = []
    for config_key in CONFIG_KEYS:
        for date in sorted(_dates_to_paths(config_key)):
            rows.append({'config_key': config_key, 'date': str(date)})
    return rows


def regrid_inputs(config_key, date):
    """Return .pp file paths for this (config_key, date)."""
    import pandas as pd
    return _inpaths_dict(_dates_to_paths(config_key)[pd.Timestamp(date)])


@rule(
    inputs=regrid_inputs,
    matrix=regrid_matrix,
    depends_on=[create_stores],
    uses={'UMProcessTasks': UMProcessTasks, 'OUTPUT_LOCATION': OUTPUT_LOCATION},
    # qos=standard rejects >1 CPU per job.
    config={'slurm': {'qos': 'high', 'mem': '100G', 'time': '10:00:00', 'cpus-per-task': 6,
                      'array_throttle': 40}},
)
def regrid(inputs, config_key, date):
    from um_to_healpix.util import load_config
    config = load_config(CONFIG_PATH)
    task = {
        'task_type': 'regrid',
        'config_path': str(CONFIG_PATH),
        'config_key': config_key,
        'date': date,
        'inpaths': list(inputs.values()),
    }
    proc = UMProcessTasks(config.processing_config[config_key], config.shared_metadata)
    proc.regrid(task)


rmk.rules_from_current_module()
