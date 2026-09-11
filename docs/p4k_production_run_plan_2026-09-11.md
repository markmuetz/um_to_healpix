# Production run plan: glm.n2560_RAL3p3_tuned_p4k with remake3 (2026-09-11)

**Status: PLANNED — not launched.** Launch only after review.

## What

Regrid and coarsen the +4K SST perturbation of `glm.n2560_RAL3p3.tuned` to HEALPix, using the remake3
pipelines (`remakefile_regrid.py`, `remakefile_coarsen.py`), writing to **prod**:
`s3://sim-data/prod/v7/glm.n2560_RAL3p3_tuned_p4k/um.{PT1H,PT3H}.hp_z{0..10}.zarr` (22 stores).

## Readiness (checked 2026-09-11)

| Item | State |
|---|---|
| Pipeline correctness | Test slice (tuned, Jan 20–31 2020, 24 dates) regridded + coarsened to z9 in `dev_remake`: **bit-identical to prod** at z10 (253 fields) and z9 (189 fields), NaN patterns identical, no coverage gaps (`docs/remake3_test_plan_2026-09-11.md`). |
| Inputs | `/work/scratch-pw6/cscullio/data/DYAMOND3/n2560_RAL3p3_tuned_p4k/glm/apver{a,b,c,d}`: **812 dates** (2020-01-20 00Z → 2021-02-28 12Z), all with exactly 4 files. Complete — no incremental runs needed. (`apvere`, 406 files, is deliberately ignored, as for tuned.) |
| Outputs | No p4k stores exist in prod, dev or dev_remake. `create_empty_zarr_stores` now refuses to overwrite existing stores. |
| Weights | Both N2560→z10 weights files exist in `/work/scratch-nopw2/mmuetz/weightsdir` (shared with tuned) — no weights generation. |
| Scope | `config.remake_config_keys = ['glm.n2560_RAL3p3_tuned_p4k']`: the remakefiles only see p4k. An unqueried `remake run` cannot touch other sims (the 11 others were produced by `um_slurm_control.py`; tuned's dev_remake test records are hidden). |
| Plan (dry run, `run -n -E slurm`) | `create_stores` ×1 → `regrid` ×812 (`--array=0-811%60`, `--dependency=afterok:<create_stores>`), qos=high, 6 CPUs, 100G, `--exclude=host1114,host1240`. Coarsen: 0 tasks until regrid succeeds (gated). |
| Compute-node submission | Verified: LOTUS compute nodes can run `sbatch` and the pixi `remake` (for the unattended coarsen trigger). |
| Home quota | ~50–100 KB per task in `.remake/` (logs, sidecars, SLURM output) → ~250 MB for 2,275 tasks. Readable per-task logs go to scratch. |

## Resources and expected timeline

Measured in the test: regrid peak RSS 82–91 GB (of 100G requested — little headroom), 27–53 min/task
(46 min mean with 24 concurrent; slow nodes up to 87 min). Coarsen z9: 2d 1.8 GB / 9 min, 3d 7.7 GB / 14 min.

| Stage | Tasks | Concurrency | Estimate |
|---|---|---|---|
| create_stores | 1 | 1 | ~30 min (37 min in test incl. weights) |
| regrid | 812 | 60 (qos=high cap: 105 at 100G) | 812/60 ≈ 14 waves × 45–60 min ≈ **10–14 h** |
| coarsen z9 | 975 (2d) + 325 (3d) | 60 per array, 32G, 12 CPUs | ~2.5 h |
| coarsen z8 → z0 | 61+16+16+16+4+1+1+1+1 (2d), 21+6+6+6+2+1+1+1+1 (3d) | chain, one zoom at a time per dim | ~2–4 h (lower zooms read more time steps per task; untested) |
| **Total** | **2,275** | | **~15–20 h** |

## Launch sequence

All from the repo root on the login node. `P=~/.pixi/bin/pixi`.

1. **Switch to prod** (the only config change left): in `config/hk26_config.py` set `deploy = 'prod'`
   (remove the TODO). Commit + push.
2. **Pre-flight**:
   ```bash
   $P run pytest tests -q                                  # 76 pass
   $P run remake info remakefile_regrid.py                 # create_stores 1 + regrid 812 pending
   $P run remake run -n -E slurm remakefile_regrid.py      # writes .remake/slurm/*.sbatch + submit.sh
   grep -E '^#SBATCH' .remake/slurm/regrid.sbatch          # array 0-811%60, qos high, exclude
   sbatch --test-only .remake/slurm/regrid.sbatch
   ```
   Check the store URL in a task's log after step 3 starts: `s3://sim-data/prod/v7/glm.n2560_RAL3p3_tuned_p4k/`.
3. **Submit regrid**: `$P run remake run -E slurm remakefile_regrid.py` → create_stores job + regrid array
   (afterok). Array job id: `.remake/jobs/regrid.jobids.json`.
4. **Submit the coarsen trigger** (unattended; plans + submits coarsen when the regrid array has finished):
   ```bash
   sbatch --dependency=afterany:<regrid array id> scripts/coarsen_after_regrid.sbatch
   ```
   (Alternative: run `$P run remake run -E slurm remakefile_coarsen.py` by hand when regrid is done.)
5. **Monitor**: `$P run remake slurm-status remakefile_regrid.py`, `remake info`, and per-task progress
   (`Completed: X/39` for regrid, `X/N` chunks for coarsen) in
   `/work/scratch-nopw2/mmuetz/um2hp/logs/prod/v7/glm.n2560_RAL3p3_tuned_p4k/`. While this session is open,
   Claude monitors the arrays (state changes, errors, progress).

## Failure handling

| Failure | Effect | Action |
|---|---|---|
| create_stores fails | regrid array never starts (afterok → DependencyNeverSatisfied) | Read `create_stores.log`; fix; `scancel` the regrid array; rerun step 3. If it failed on "Refusing to recreate existing zarr stores", something already wrote stores — investigate before deleting anything. |
| Some regrid tasks fail (OOM at ~91/100G, timeout, S3 error, bad node) | Others continue; coarsen trigger still runs (afterany) and coarsens only fully-regridded batches | After the array leaves the queue: `remake info -F`; fix (e.g. bump `mem`, add node to `slurm_config['exclude']`); `remake run -E slurm remakefile_regrid.py` resubmits only failed/pending tasks; then rerun coarsen to pick up the rest. |
| Slow node | Tasks 3–5× slower (host1114, host1240 seen) | Add to `exclude` (config, untracked → no reruns); affects the next submission only; `scancel` stragglers and resubmit if needed. |
| A coarsen task fails | Downstream zoom rules for that dim never start (rule-level afterok) | `scancel` the stuck arrays; `remake run -E slurm remakefile_coarsen.py` resubmits failed + downstream. |
| Home quota | Tasks fail to record results (logged earlier today) | Keep headroom; the run adds ~250 MB. |
| Input purge | Inputs are on another user's scratch (`scratch-pw6/cscullio`) | Launch soon; failed input reads show as task failures. |

Rollback (if outputs are wrong): delete the p4k prod stores on S3, mark the tasks pending
(`remake set-state ... --pending`), fix, rerun from step 3.

## Verification after the run

1. `remake info` on both remakefiles: 813 + 1,462 tasks up to date, 0 to run.
2. Coverage and plausibility vs the control (tuned, prod) at z10 and a low zoom:
   ```bash
   $P run python scripts/compare_stores.py glm.n2560_RAL3p3_tuned_p4k --a prod --b prod \
       --b-key glm.n2560_RAL3p3.tuned --zoom 10 --times 2020-03-01T12 2020-07-01T12 2021-02-28T12
   ```
   Expect: same variables/coords, **no coverage gaps, full time range** (to 2021-03-01T00), identical NaN
   patterns (e.g. `mrsol` ocean mask); value differences everywhere (flags expected), with plausible
   `meanDiff` (e.g. `tas`, `ta` a few K warmer). Repeat at `--zoom 5` and `--zoom 0` for the coarsening chain.
3. Spot-check a few fields visually (easygems / intake) before announcing.

## Not in scope / known limitations

- No rerun propagation from regrid to coarsen: if a regrid task is re-run after coarsening, force the affected
  coarsen batches (`--force -Q 'start ...'`).
- `regrid`'s `uses=` includes all of `UMProcessTasks`: editing any method (even coarsen code) marks all 812
  regrid tasks stale. Don't edit the class mid-run; if needed, re-stamp with `set-state --success`.
