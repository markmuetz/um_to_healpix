# Remake3 Migration Plan

## Overview

Replace `um_slurm_control.py` and `um_process_tasks.py` with two remake3 remakefiles, matching the current two-phase workflow (`process` then `coarsen`). The core processing modules (`latlon_to_healpix.py`, `healpix_coarsen.py`, `cube_to_da_mapping.py`) and config (`hk26_config.py`) stay unchanged.

## Why remake3 is a good fit

- The pipeline is a clear three-stage DAG: create → regrid → coarsen
- Tasks are parameterized over (config_key, date, zoom, dim) — maps directly to remake3 matrices
- SLURM array jobs with dependency chains are exactly what remake3's SLURM executor provides
- Python config with iris Constraints and callables works naturally (just import the config)
- Change detection via AST comparison means config or processing code changes automatically trigger reruns of affected tasks

## Two remakefiles

The current workflow has two separate CLI commands (`um-slurm-control process` and `um-slurm-control coarsen`) run at different times. We preserve this with two remakefiles:

- `remakefile_regrid.py` — create stores + regrid (run when .pp files are available)
- `remakefile_coarsen.py` — coarsen zoom chain (run after all regridding is complete)

This avoids the problem of coarsening starting before all regrid tasks are done. The regrid matrix is built by scanning whatever .pp files exist at plan time, and the user triggers coarsening manually once regridding is confirmed complete.

### remakefile_regrid.py

Two rules: `create_stores` and `regrid`.

```python
from remake import Remake, rule
from um_to_healpix.config.hk26_config import CONFIG

rmk = Remake(config={'slurm': {'partition': 'standard', 'time': '02:00:00', 'mem': '16G'}})

CONFIG_KEYS = list(CONFIG.keys())

def create_inputs(config_key):
    """Return first-date .pp paths for this config_key."""
    cfg = CONFIG[config_key]
    # scan logic currently in um_slurm_control.process()
    ...

@rule(
    inputs=create_inputs,
    matrix={'config_key': CONFIG_KEYS},
    config={'slurm': {'mem': '32G'}},
)
def create_stores(inputs, config_key):
    cfg = CONFIG[config_key]
    # call existing create_empty_zarr_stores logic
    ...

def regrid_matrix():
    """Scan input directories, return (config_key, date) pairs for available .pp files."""
    rows = []
    for config_key, cfg in CONFIG.items():
        for date in scan_dates(cfg):
            rows.append({'config_key': config_key, 'date': date})
    return rows

def regrid_inputs(config_key, date):
    """Return .pp file paths for this (config_key, date)."""
    cfg = CONFIG[config_key]
    ...

@rule(
    inputs=regrid_inputs,
    matrix=regrid_matrix,
    depends_on=[create_stores],
    config={'slurm': {'cpus': 6, 'mem': '32G', 'time': '02:00:00'}},
)
def regrid(inputs, config_key, date):
    cfg = CONFIG[config_key]
    # call existing regrid logic from UMProcessTasks.regrid()
    ...

rmk.rules_from_current_module()
```

Usage:
```bash
remake run remakefile_regrid.py -E slurm
```

Can be rerun as more .pp files land — remake3 skips completed tasks.

### remakefile_coarsen.py

One rule per zoom level, generated in a loop. Each depends on the previous zoom (or on regrid being complete, which the user ensures manually).

```python
from remake import Remake, rule
from um_to_healpix.config.hk26_config import CONFIG

rmk = Remake(config={'slurm': {'partition': 'standard', 'time': '01:00:00', 'mem': '16G'}})

MAX_ZOOM = 10

def make_coarsen_matrix(zoom):
    """Return (config_key, dim, time_batch) tuples for this zoom level."""
    rows = []
    for config_key, cfg in CONFIG.items():
        for dim in ['2d', '3d']:
            for time_batch in get_time_batches(cfg, dim, zoom):
                rows.append({'config_key': config_key, 'dim': dim, 'time_batch': time_batch})
    return rows

coarsen_rules = {}
for zoom in range(MAX_ZOOM - 1, -1, -1):  # 9, 8, ..., 0
    upstream = [coarsen_rules[zoom + 1]] if (zoom + 1) in coarsen_rules else []

    @rule(
        matrix=make_coarsen_matrix(zoom),
        depends_on=upstream,
        config={'slurm': {'cpus': 12, 'mem': '16G'}},
    )
    def coarsen_zN(config_key, dim, time_batch, _zoom=zoom):
        cfg = CONFIG[config_key]
        # call existing coarsen_healpix_zarr_region logic
        ...

    coarsen_zN.__name__ = f'coarsen_z{zoom}'
    coarsen_rules[zoom] = coarsen_zN

rmk.rules_from_current_module()
```

This produces the chain: `coarsen_z9 → coarsen_z8 → ... → coarsen_z0`. Tasks within each zoom level run in parallel via SLURM array jobs.

Usage:
```bash
# Run after regridding is confirmed complete
remake run remakefile_coarsen.py -E slurm
```

## What changes

| Component | Fate |
|---|---|
| `um_slurm_control.py` | **Replaced** by remakefiles + `remake run -E slurm` |
| `um_process_tasks.py` | **Replaced** — rule functions call core logic directly |
| `latlon_to_healpix.py` | Unchanged |
| `healpix_coarsen.py` | Unchanged |
| `cube_to_da_mapping.py` | Unchanged |
| `config/hk26_config.py` | Unchanged |
| `um_parse_slurm.py` | **Keep** — independent sacct/squeue utility |
| Done-marker files | **Removed** — remake3 tracks task completion |
| SLURM script templates | **Removed** — remake3's SLURM executor handles this |

## Implementation steps

1. Prototype the coarsen zoom chain first — this is the most novel mapping and needs validation that remake3 handles loop-generated rules correctly.
2. Write `remakefile_regrid.py`, extracting the scan and regrid logic from `um_slurm_control.py` / `um_process_tasks.py` into standalone functions.
3. Write `remakefile_coarsen.py` with the per-zoom rule loop.
4. Test locally with `remake run` on a small subset (one config_key, a few dates, two zoom levels).
5. Test on SLURM with `remake run -E slurm`.
6. Remove `um_slurm_control.py`, `um_process_tasks.py`, and done-marker infrastructure once validated.

## Risks

- **Loop-generated rules**: The coarsen zoom chain uses `__name__` renaming and closure-captured zoom values. Needs validation that remake3 registers these correctly and that AST-based change detection works on them.
- **SLURM throttling**: Multiple tasks write regions of shared zarr stores. Check that remake3's SLURM executor supports array job throttling (`%N`) to avoid S3 contention.
- **Input scanning at plan time**: The regrid matrix scans .pp file directories. `remake run` planning must happen on JASMIN where the data is visible.

## Status / findings (2026-09-11)

First test run of the remakefiles. Details, commands and a progress log are in `remake3_test_plan_2026-09-11.md`.

### Risks resolved
- **Loop-generated rules**: register correctly as `coarsen_z9 → … → coarsen_z0` (checked with `remake rule-dag -N`). AST change detection on them is not yet verified.
- **SLURM throttling**: supported via the per-rule `array_throttle` slurm config key (`--array=0-N%T`).
- **Input scanning at plan time**: fast. A filtered regrid plan takes about 5 s.

### New findings (fixed on this branch)
- **`uses=` values are hashed by `repr()`**: `PROCESSING_CONFIG`'s repr contains function addresses (`at 0x7f…`), so every task would rerun on every invocation. The config is now loaded inside the rule bodies and is not tracked. Only `deploy`/`output_vn` go in `uses=`, so switching output location reruns tasks.
- **SLURM keys are written verbatim** as `#SBATCH --<key>=<value>`, so the key must be `cpus-per-task`, not `cpus_per_task`.
- **qos**: `standard` rejects >1 CPU per job (`QOSMaxCpuPerNode`), so `regrid`/`coarsen` need `qos=high` (as the old code used).
- **Coarsen `donepath`**: `coarsen_healpix_region` required a `donepath` per subtask. It's now optional.
- **Weights**: `_gen_orog_land_sea` asserted that the weights existed but ran before weight generation. It now generates them if missing (weightsdir had been purged).
- **kscale GWS moved** to `/gws/ssde/j25b/kscale`.

### Deviations from the plan above
- The rule bodies still call `UMProcessTasks` and `find_dyamond3_pp_dates_to_paths`. Before step 6 (delete `um_slurm_control.py`/`um_process_tasks.py`), that logic must move into standalone functions.
- Step 4 (local test) was skipped: an N2560 regrid needs about 100 GB and 6 CPUs, so testing went straight to SLURM on a small slice.
- `config/hk26_config.py` is not unchanged: it has the path fix, an optional per-sim `pp_glob`, and the new `glm.n2560_RAL3p3_tuned_p4k` sim.
- The environment is pixi (`[tool.pixi.*]` in `pyproject.toml`), not conda.
