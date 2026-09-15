"""Remake3 remakefile for create_empty_zarr_stores and regrid tasks.

Run when .pp files are available (from the repo root, in the pixi env):
    pixi run remake run remakefile_regrid.py -E slurm

Can be rerun as more .pp files land — remake3 skips completed tasks.

The processing config is deliberately *not* tracked for reruns (its repr embeds memory addresses) -
it is loaded inside each rule. Only the output location (deploy/output_vn) is tracked.
Rerun after config edits with --force -Q.

Input discovery: the regrid matrix scans each sim's input dir at plan time and writes an index
(config.pp_indexdir/<config_key>.json). SLURM array elements rebuild their task's inputs from that
index instead of rescanning the (GWS) input dirs.

Per-task logs are also written (appended) to config.logdir/<config_key>/.
"""
import json
from pathlib import Path

import pandas as pd
from loguru import logger
from remake import Remake, rule

from um_to_healpix.um_process_tasks import UMProcessTasks
from um_to_healpix.pp_scan import DEFAULT_PP_GLOB, find_dyamond3_pp_dates_to_paths
from um_to_healpix.util import load_config

CONFIG_PATH = Path('config/hk26_config.py')
config_module = load_config(CONFIG_PATH)
PROCESSING_CONFIG = config_module.processing_config
# Only the sims this pipeline is responsible for (config.remake_config_keys; default all).
CONFIG_KEYS = list(getattr(config_module, 'remake_config_keys', PROCESSING_CONFIG.keys()))
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
        # Bad nodes, from the (untracked) config; empty -> no --exclude line.
        'exclude': config_module.slurm_config.get('exclude', ''),
    },
})

# config_key -> {pd.Timestamp: [paths]}, filled by a scan (plan time) or from the index (array elements).
_DATES_TO_PATHS = {}
# config_keys scanned in this process (so a process that plans more than once scans only once).
_SCANNED = set()


def _index_path(config_key):
    return config_module.pp_indexdir / f'{config_key}.json'


def _read_index(index_path):
    if not index_path.exists():
        return {}
    index = json.loads(index_path.read_text())
    return {pd.Timestamp(d): [Path(p) for p in paths] for d, paths in index.items()}


def _scan_dates_to_paths(config_key):
    """Scan the input dir for this config_key and (re)write its index. Plan time only."""
    cfg = PROCESSING_CONFIG[config_key]
    dates_to_paths = find_dyamond3_pp_dates_to_paths(cfg['basedir'], cfg.get('pp_glob', DEFAULT_PP_GLOB))
    index_path = _index_path(config_key)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not dates_to_paths:
        # A vanished input dir otherwise looks exactly like "nothing to do": the index is silently emptied, the
        # rule plans 0 tasks, every completed task drops out of `remake info`, and the coarsen rules (which gate
        # on regrid-done dates) cascade to 0 as well. `remake run` then prints "Nothing to do" and exits 0.
        # Happened for real on 2026-09-14 when the p4k .pp source was deleted (index rebuilt from the task logs).
        # Keep the index so a finished run stays inspectable; a *run* still fails, on the missing input paths.
        previous = _read_index(index_path)
        if previous:
            logger.warning(f'scan of {cfg["basedir"]} found no inputs for {config_key}, but its index lists '
                           f'{len(previous)} dates - keeping the index. The inputs have been deleted or moved; '
                           f'completed tasks stay visible, but nothing can be re-run until they are restored.')
            _DATES_TO_PATHS[config_key] = previous
            _SCANNED.add(config_key)
            return previous
        logger.warning(f'scan of {cfg["basedir"]} found no inputs for {config_key}')
    index = {str(date): [str(p) for p in paths] for date, paths in sorted(dates_to_paths.items())}
    tmp_path = index_path.with_suffix(f'.{config_key}.tmp')
    tmp_path.write_text(json.dumps(index, indent=1))
    tmp_path.replace(index_path)
    _DATES_TO_PATHS[config_key] = dates_to_paths
    _SCANNED.add(config_key)
    return dates_to_paths


def _dates_to_paths(config_key):
    """This process's view of the inputs: scanned this process, else the plan-time index, else a scan."""
    if config_key not in _DATES_TO_PATHS:
        index = _read_index(_index_path(config_key))
        if index:
            _DATES_TO_PATHS[config_key] = index
        else:
            _scan_dates_to_paths(config_key)
    return _DATES_TO_PATHS[config_key]


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
    # Generates N2560 z10 weights (30-40G each) if they do not exist. Peak RSS 39.9G in the p4k run.
    config={'slurm': {'mem': '64G', 'time': '24:00:00'}},
)
def create_stores(inputs, config_key):
    """Create empty zarr stores for all zooms. Refuses to overwrite existing stores (would delete data)."""
    from um_to_healpix.util import load_config, task_log
    config = load_config(CONFIG_PATH)
    task = {
        'task_type': 'create_empty_zarr_stores',
        'config_path': str(CONFIG_PATH),
        'config_key': config_key,
        'inpaths': list(inputs.values()),
    }
    with task_log(config.logdir / config_key / 'create_stores.log'):
        proc = UMProcessTasks(config.processing_config[config_key], config.shared_metadata)
        proc.create_empty_zarr_stores(task)


def regrid_matrix():
    """Scan input directories (writing the index), return (config_key, date) pairs for available .pp files."""
    rows = []
    for config_key in CONFIG_KEYS:
        dates_to_paths = _DATES_TO_PATHS[config_key] if config_key in _SCANNED else _scan_dates_to_paths(config_key)
        for date in sorted(dates_to_paths):
            rows.append({'config_key': config_key, 'date': str(date)})
    return rows


def regrid_inputs(config_key, date):
    """Return .pp file paths for this (config_key, date)."""
    return _inpaths_dict(_dates_to_paths(config_key)[pd.Timestamp(date)])


@rule(
    inputs=regrid_inputs,
    matrix=regrid_matrix,
    depends_on=[create_stores],
    uses={'UMProcessTasks': UMProcessTasks, 'OUTPUT_LOCATION': OUTPUT_LOCATION},
    # qos=standard rejects >1 CPU per job.
    # mem: measured peak is 60.1G with the per-step model-level interpolation (was 98.5G over 812 p4k tasks,
    # where 100G left no headroom and tasks thrashed in memory reclaim on packed nodes, losing 24 to
    # walltime/OOM). 96G is ~1.6x the peak and, just as importantly, caps SLURM at 16 of our tasks per 1.5TB
    # node: the request is the only lever on packing density, and density drove duration in the p4k run
    # (3-4 tasks/node 60 min, 8+ tasks/node 104 min). See docs/p4k_production_run_plan_2026-09-11.md.
    # array_throttle: 45 measured best (49 tasks/h vs 22 at 30 and 39-41 at 60).
    config={'slurm': {'qos': 'high', 'mem': '96G', 'time': '10:00:00', 'cpus-per-task': 6,
                      'array_throttle': 45}},
)
def regrid(inputs, config_key, date):
    import pandas as pd
    from um_to_healpix.util import load_config, task_log
    config = load_config(CONFIG_PATH)
    task = {
        'task_type': 'regrid',
        'config_path': str(CONFIG_PATH),
        'config_key': config_key,
        'date': date,
        'inpaths': list(inputs.values()),
    }
    log_name = f'{pd.Timestamp(date):%Y%m%dT%H}.log'
    with task_log(config.logdir / config_key / 'regrid' / log_name):
        proc = UMProcessTasks(config.processing_config[config_key], config.shared_metadata)
        proc.regrid(task)


rmk.rules_from_current_module()
