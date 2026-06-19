"""Remake3 remakefile for create_empty_zarr_stores and regrid tasks.

Run when .pp files are available:
    remake run remakefile_regrid.py -E slurm

Can be rerun as more .pp files land — remake3 skips completed tasks.
"""
import math
from pathlib import Path

from remake import Remake, rule

from um_to_healpix.um_slurm_control import find_dyamond3_pp_dates_to_paths
from um_to_healpix.um_process_tasks import UMProcessTasks, slurm_run
from um_to_healpix.util import load_config

CONFIG_PATH = Path('config/hk26_config.py')
config_module = load_config(CONFIG_PATH)
PROCESSING_CONFIG = config_module.processing_config
CONFIG_KEYS = list(PROCESSING_CONFIG.keys())

rmk = Remake(config={
    'slurm': {
        'account': config_module.slurm_config['account'],
        'partition': 'standard',
        'qos': 'standard',
        'time': '10:00:00',
        'mem': '100G',
    },
})


def create_inputs(config_key):
    cfg = PROCESSING_CONFIG[config_key]
    dates_to_paths = find_dyamond3_pp_dates_to_paths(cfg['basedir'])
    first_date = cfg['first_date']
    if first_date not in dates_to_paths:
        raise ValueError(f'First date {first_date} not found for {config_key}')
    return {'inpath_' + str(i): str(p) for i, p in enumerate(dates_to_paths[first_date])}


@rule(
    inputs=create_inputs,
    matrix={'config_key': CONFIG_KEYS},
    uses={'PROCESSING_CONFIG': PROCESSING_CONFIG, 'UMProcessTasks': UMProcessTasks},
    config={'slurm': {'mem': '100G', 'time': '10:00:00'}},
)
def create_stores(inputs, config_key):
    cfg = PROCESSING_CONFIG[config_key]
    inpaths = list(inputs.values())
    task = {
        'task_type': 'create_empty_zarr_stores',
        'config_path': str(CONFIG_PATH),
        'config_key': config_key,
        'inpaths': inpaths,
    }
    proc = UMProcessTasks(cfg, config_module.shared_metadata)
    proc.create_empty_zarr_stores(task)


def regrid_matrix():
    """Scan input directories, return (config_key, date) pairs for available .pp files."""
    rows = []
    for config_key, cfg in PROCESSING_CONFIG.items():
        dates_to_paths = find_dyamond3_pp_dates_to_paths(cfg['basedir'])
        for date in sorted(dates_to_paths.keys()):
            rows.append({'config_key': config_key, 'date': str(date)})
    return rows


def regrid_inputs(config_key, date):
    """Return .pp file paths for this (config_key, date)."""
    import pandas as pd
    cfg = PROCESSING_CONFIG[config_key]
    dates_to_paths = find_dyamond3_pp_dates_to_paths(cfg['basedir'])
    ts = pd.Timestamp(date)
    paths = dates_to_paths[ts]
    return {'inpath_' + str(i): str(p) for i, p in enumerate(paths)}


@rule(
    inputs=regrid_inputs,
    matrix=regrid_matrix,
    depends_on=[create_stores],
    uses={'PROCESSING_CONFIG': PROCESSING_CONFIG, 'UMProcessTasks': UMProcessTasks},
    config={'slurm': {'mem': '100G', 'time': '10:00:00', 'cpus_per_task': 6}},
)
def regrid(inputs, config_key, date):
    import pandas as pd
    cfg = PROCESSING_CONFIG[config_key]
    inpaths = list(inputs.values())
    task = {
        'task_type': 'regrid',
        'config_path': str(CONFIG_PATH),
        'config_key': config_key,
        'date': date,
        'inpaths': inpaths,
    }
    proc = UMProcessTasks(cfg, config_module.shared_metadata)
    proc.regrid(task)


rmk.rules_from_current_module()
