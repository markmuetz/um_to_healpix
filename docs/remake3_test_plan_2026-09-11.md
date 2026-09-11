# Plan: test remake3 orchestration, then process n2560_RAL3p3_tuned_p4k (2026-09-11)

## Context
Branch `remake3_migration` adds `remakefile_regrid.py` and `remakefile_coarsen.py`, which replace `um_slurm_control.py`, `um_process_tasks.py` and the `.done` files. Neither remakefile has been run yet. Today has two goals:
1. Test the remakefiles end-to-end on a small slice: `glm.n2560_RAL3p3.tuned`, 2020-01-20 → 2020-01-31 (24 × 12-hourly `.pp` dates), steps create → regrid (z10) → coarsen z10→z9. Output goes to `s3://sim-data/dev_remake/v7/...` and you check it.
2. Once you've validated the test output, process the new model at `/work/scratch-pw6/cscullio/data/DYAMOND3/n2560_RAL3p3_tuned_p4k` with remake (decided: remake, gated on your check).

## What I found (blockers)
| # | Issue | Evidence |
|---|---|---|
| 1 | The kscale GWS moved: `/gws/nopw/j04/kscale` → `/gws/ssde/j25b/kscale`. `dy3dir` and `orig_base_dir` in `config/hk26_config.py` are stale, so global sims find no inputs. | `ls` fails on the old path; the new path has `DYAMOND3_reruns/5km-RAL3p3-tuned/glm/field.pp/apver{a..e}.pp`, with 24 Jan files per stream |
| 2 | `uses={'PROCESSING_CONFIG': ...}` is hashed by `repr()`, which contains `at 0x7f…`, so every task would rerun on every `remake run`. | The repr sha1 differed between two processes. `remake/core/scope.py:uses_parts` uses `repr()` for non-callables |
| 3 | `config={'slurm': {'cpus_per_task': N}}` is written verbatim as `#SBATCH --cpus_per_task=N`, which sbatch rejects. | `sbatch: unrecognized option '--cpus_per_task=2'`; remake `slurm_executor.py:82` uses keys as-is |
| 4 | qos `standard` with >1 CPU gets rejected (`QOSMaxCpuPerNode`). The old code used `qos=high` for regrid and coarsen; the regrid remakefile uses `standard`. | `sbatch --test-only` |
| 5 | `coarsen_healpix_region` does `Path(subtask['donepath'])`, but the remakefile's `tgt_times` has no `donepath`, so coarsen raises `KeyError` on the first subtask. | `um_process_tasks.py:567` |
| 6 | `weightsdir` (`/work/scratch-nopw2/mmuetz/weightsdir`) is empty (it looks purged). `_gen_orog_land_sea` asserts that weights exist, and it runs *before* the template step that generates weights, so `create_stores` would fail. | `um_process_tasks.py:267`, called from line 397 before line 412 |
| 7 | No env has remake3 plus the project deps. `um2hp_env`'s `um_to_healpix` is an editable install pointing at `~/deploy`, so jobs would run the deploy code. | `pip`/import checks |
| 8 | The p4k layout is `glm/apver*/*.pp` (no `field.pp/`), so `find_dyamond3_pp_dates_to_paths`' glob `field.pp/apve*/**/*.pp` misses it. | `ls` |

Checked and fine: loop-generated `coarsen_z*` rules register correctly. Filtered regrid planning takes about 5 s. `~/.s3cfg` exists. The queue is idle. `~/projects/remake` fast-forwarded to origin (0.8.3).

## Step -1: docs
- Save this plan as `docs/remake3_test_plan_2026-09-11.md` and update it as we go, with job IDs, outcomes and any deviations.
- Add a **Status / findings (2026-09-11)** section to `docs/remake3_migration_plan.md`:
  - Risks resolved: loop-generated rules register correctly (`coarsen_z9…z0`), throttling is covered by `array_throttle`, and plan-time scanning is fast.
  - New findings: blockers 2–5 above (uses/repr, `cpus-per-task`, qos, donepath).
  - Deviations:
    - The rule bodies still call `UMProcessTasks` and `find_dyamond3_pp_dates_to_paths`, so implementation step 6 (delete the old modules) first needs that logic moved out.
    - Step 4 (local test) is skipped: N2560 regrid needs 100 GB and 6 CPUs, so we go straight to SLURM on a small slice.
    - "`config/hk26_config.py` unchanged" no longer holds (path fix, `pp_glob`, p4k).

## Step 0: environment (pixi)
pixi 0.71.2 is installed at `~/.pixi/bin/pixi`, which isn't on PATH here, so I'll call it by full path. Its cache is on scratch (`/work/scratch-nopw2/mmuetz/pixi-cache`, `netfs-redirect = "never"`).
- Put the manifest in `pyproject.toml` as `[tool.pixi.workspace]` (conda-forge, linux-64), so everything stays in one file:
  - `[tool.pixi.dependencies]`: the conda deps from `envs/um2hp_env.yml` (python 3.12, iris, cartopy, esmpy, mo_pack, python-stratify, s3fs, dask/distributed, zarr, xarray, `easygems 0.0.14`, loguru, click, healpy, ...).
  - `[tool.pixi.pypi-dependencies]`: `um_to_healpix = { path = ".", editable = true }` and `remake = { path = "../../remake", editable = true }`. The editable remake means remake bugs found today can be fixed in place. Later I'd switch to a git rev.
  - `[tool.pixi.tasks]`: `test = "pytest tests/"`.
  - Commit `pixi.lock`.
- `pixi install`. The env lands in `.pixi/` inside the repo (home), about 3–5 GB, copied from the scratch cache. Add `.pixi/` and `.remake/` to `.gitignore`. Remove the stray `.remake/` that my `rule-dag` probe created; it holds an empty DB only.
- The freshly resolved versions will be newer than prod `um2hp_env` (the pyproject already requires `zarr>=3.2.1` and `distributed>=2026.3`). `pixi run test` must pass before any submission, and your check of the test output also covers the version change.
- **Submission:** always `pixi run remake run ... -E slurm` from the repo root. `pixi run` puts `.pixi/envs/default/bin` (and the activation env vars) on PATH, and SLURM `--export=ALL` carries them to the jobs, so `remake run-array-task` resolves. Verify this in the first job's `.out`.

## Step 1: code and config changes (branch `remake3_migration`)
**`config/hk26_config.py`**
- `deploy = 'dev_remake'` (for the test only; set it back to `'prod'` for step 5).
- `dy3dir` and `orig_base_dir` → `/gws/ssde/j25b/kscale/...`.
- Add `glm.n2560_RAL3p3_tuned_p4k` to `global_configs` as a copy of the `glm.n2560_RAL3p3.tuned` entry. Override `name`, `basedir` (`.../n2560_RAL3p3_tuned_p4k/glm`), `pp_glob='apve*/*.pp'`, `donepath_tpl`/`coarsen_donepath_tpl`/`zarr_store_url_tpl` (all key-based), and `metadata` (`simulation` = key, plus the tuned description with an added "+4K SST" sentence for you to edit). The orog, groups and `max_zoom=10` are the same as tuned (same grid, so the weights are shared).

**`src/um_to_healpix/um_slurm_control.py`**: add a `pp_glob='field.pp/apve*/**/*.pp'` parameter to `find_dyamond3_pp_dates_to_paths`. Existing callers are unchanged.

**`src/um_to_healpix/um_process_tasks.py`**
- `_gen_orog_land_sea`: if the weights file is missing, call `gen_weights(land, weights_path, zoom=max_zoom, add_cyclic=..., regional=...)` instead of asserting (reusing `latlon_to_healpix.gen_weights`).
- `coarsen_healpix_region`: write the donepath only when the subtask has `'donepath'` in it. The old pipeline keeps working.

**`remakefile_regrid.py`**
- Drop `PROCESSING_CONFIG` from `uses` (your choice: no config tracking). Rule bodies load the config locally (`from um_to_healpix.util import load_config; load_config(CONFIG_PATH)`), so it's neither a free global nor hashed.
- Keep `UMProcessTasks` in `uses`. Add `'DEPLOY': config_module.deploy, 'OUTPUT_VN': config_module.output_vn`. These are stable strings, and they stop the dev_remake records from marking prod tasks as done when `deploy` flips. I'm flagging this in case you'd rather not have it.
- Cache the directory scan (`functools.cache` keyed on `(basedir, pp_glob)`), so `regrid_inputs` doesn't re-glob for every task. Pass `cfg.get('pp_glob', ...)`.
- SLURM config:
  - `qos='high'` for `regrid`, `cpus-per-task: 6` (hyphens)
  - `create_stores` gets `time='24:00:00'`, because it now generates about 3–4 N2560 z10 weight files at 30–40 GB each
  - `'export': 'ALL,OMP_NUM_THREADS=1'` (what the old sbatch template did)
  - `array_throttle: 40`
- Remove the unused `math` and `slurm_run` imports.

**`remakefile_coarsen.py`**: the same `uses` fix (load the config locally, keep `zoom`, `NBATCH`, `DEPLOY`, `OUTPUT_VN`), `cpus-per-task`, `qos='high'`, the `export` setting, and remove the unused `math`/`batched` imports.

**Tests**: add unit tests for the `pp_glob` scan (tmp dir with both layouts) and for coarsen with no `donepath`. Run `pytest tests/`.

## Step 2: static checks
```
remake lint remakefile_regrid.py; remake lint remakefile_coarsen.py
remake run -n remakefile_regrid.py -Q 'config_key == "glm.n2560_RAL3p3.tuned" and (rule == "create_stores" or date < "2020-02-01")'   # expect 1 + 24
```
Check the generated `.remake/slurm/*.sbatch` for correct `#SBATCH` lines. Then `sbatch --test-only` it.

## Step 3: test run (you review each submission)
Let `Q='config_key == "glm.n2560_RAL3p3.tuned" and (rule == "create_stores" or date < "2020-02-01")'`.
1. `remake run remakefile_regrid.py -E slurm -Q "$Q"`: create_stores (weights + empty stores for z10..z0), then 24 regrid tasks with an afterok dependency.
2. Monitor with `remake slurm-status` and `remake info --json`. Triage failures with `remake info -F` and `sacct`.
3. Coarsen z10→z9 only, for complete data (2d: hours 0–279, 3d: 3-hourly steps 0–89):
   `remake run remakefile_coarsen.py -E slurm -Q 'rule == "coarsen_z9" and config_key == "glm.n2560_RAL3p3.tuned" and ((dim == "2d" and batch_id < 28) or (dim == "3d" and batch_id < 9))'` (28 + 9 tasks)

## Step 4: verification (before handing over)
- Rerun both `remake run -n ... -Q ...` and expect **0 tasks**. This proves the `uses` fix, and `remake why` gives the reason for any task that would still run.
- Open `s3://sim-data/dev_remake/v7/glm.n2560_RAL3p3.tuned/um.PT1H.hp_z10.zarr` and `..._z9.zarr` (and PT3H), and check that each variable has non-NaN data for Jan 20–31 and NaN afterwards. Also check that z9 equals the mean of 4 z10 cells for one timestep.
- Then hand over to you for your own testing.

## Step 5: p4k via remake (after you sign off)
- Set `deploy='prod'` in the config (with `output_vn` staying `v7`; confirm then). Plan with `-Q 'config_key == "glm.n2560_RAL3p3_tuned_p4k"'`, i.e. 1 create + 812 regrid (`array_throttle` 40). `-n` first, then `-E slurm`.
- Rerun regrid with `remake run` (it's idempotent) as tasks fail or finish. Coarsen the full z9→z0 chain later, once regrid is complete. That's outside today's scope.
- The weights are already generated by step 3 (same grid), so p4k's `create_stores` is quick. Don't run the two `create_stores` at the same time (weights-file race).

## Follow-ups (not today)
- The kscale path fix and the p4k config also need to reach `main` and `~/deploy`.
- Coarsen has no link to regrid completion. That's acceptable for now: it's the manual gate from the migration plan.

## Progress log

| Time | Step | Outcome |
|---|---|---|
| 2026-09-11 | -1 docs | Plan saved here; status/findings section added to `remake3_migration_plan.md`. |
| 2026-09-11 | 0 env | pixi manifest in `pyproject.toml`; `pixi install` OK (1.5G `.pixi/`). First solve picked s3fs 0.4.2 because `botocore = "*"` was unconstrained → dropped botocore, pinned `s3fs >=2025.1` (now 2026.7.0). Versions: zarr 3.3.0, xarray 2026.7.0, dask 2026.8.0, iris 3.16.0, numpy 2.5.3, easygems 0.0.14, remake 0.8.3 (editable). Baseline `pixi run test`: 68 passed. **Note:** prod `um2hp_env` has easygems 0.1.1 despite the 0.0.14 pin. |
| 2026-09-11 | 1 code | Config (deploy=dev_remake, kscale path, p4k entry + `pp_glob`), `find_dyamond3_pp_dates_to_paths(pp_glob=...)`, orog weights generated if missing, optional coarsen donepath, remakefiles rewritten. New `tests/unit/test_remake_support.py`; 73 passed. |
| 2026-09-11 | 2 static | `remake lint` clean (no scope warnings). Regrid sees 12 sims / 9743 dates. Dry run of test slice: 1 create_stores + 24 regrid. Generated sbatch correct (`--cpus-per-task=6`, `--qos=high`, `--array=0-23%40`, `--export=ALL,OMP_NUM_THREADS=1`, afterok dependency); both pass `sbatch --test-only`. |
| 2026-09-11 12:06 | 3.1 submit | Committed/pushed `4bb604c`. Submitted test slice: create_stores job **51369955**, regrid job **51369956** (`0-23%40`, afterok). Task log created on compute node → `remake` resolved via pixi PATH + `--export=ALL`. |
| 2026-09-11 12:43 | 3.1 | create_stores COMPLETED (37 min; weights ~5 min/grid, 2 grids; peak RSS 39.9G of 100G). Regrid array started. |
| 2026-09-11 13:10 | 3.1 | Home quota hit (127G) while jobs ran: elements 5 and 21 lost some log lines (loguru "logging error", non-fatal). Space freed by user; no task failures. |
| 2026-09-11 13:30 | design | Redesign after review (no coarsen task had run, so keys could change freely): (1) coarsen matrix gated on *successful* regrid dates, read via the remake Python API (`load_remake` + `status_summary`/`iter_tasks`), requiring files floor(t,12h) and floor(t-1h,12h); (2) independent per-dim chains `coarsen_{2d,3d}_z9..z0` using `@rule(name=...)`; (3) batches keyed by `start` (ISO time) instead of `batch_id`; (4) per-task logs appended to `logdir` on scratch; (5) plan-time .pp index on scratch read by array elements (no per-element rescans); (6) `create_empty_zarr_stores` refuses to overwrite existing stores (any rerun of create_stores, e.g. after a `UMProcessTasks` edit, would have wiped all regridded data). 76 tests pass. |
| 2026-09-11 14:15 | 3.1 | Regrid elements 3, 4, 22, 23 cancelled (22: slow node host1114; others: repo moved to ~/projects/local mid-run, env rebuilt). Successful tasks re-stamped (`set-state --success`, succeeded dates only); 4 resubmitted (51403735) with `--exclude`. host1240 also slow (~3x) → excluded. |
| 2026-09-11 15:09 | 3.1 | Regrid test slice complete: 25/25 up to date, `run -n` plans nothing (uses= repr fix confirmed across runs). |
| 2026-09-11 15:25 | 3.2 | coarsen_2d_z9 (28 batches, 51413584) + coarsen_3d_z9 (9 batches, 51413585) all COMPLETED, ~10-15 min each. z9 coverage exactly as gated: PT1H to 31T15, PT3H to 31T03. |
| 2026-09-11 15:40 | 3.3 | `scripts/compare_stores.py` dev_remake vs prod: z10 253 fields (all 39 vars, 4 times, 3 levels) and z9 189 fields: all **bit-identical**, NaN patterns identical (mrsol ocean NaNs match). z10 coverage complete (no gaps) for all vars. |
