# Production run plan: glm.n2560_RAL3p3_tuned_p4k with remake3 (2026-09-11)

**Status: LAUNCHED 2026-09-11** (approved after review). Monitoring: `scripts/watch_remake.py` (ntfy topic `hk26-jasmin-updates`).

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
4. **Coarsen: submitted manually** once regrid is complete (decision 2026-09-11 22:08: the trigger job 51462412 was cancelled). `$P run remake run -E slurm remakefile_coarsen.py`. The trigger option, for reference:
   ```bash
   sbatch --dependency=afterany:<regrid array id> scripts/coarsen_after_regrid.sbatch
   ```
   (Alternative: run `$P run remake run -E slurm remakefile_coarsen.py` by hand when regrid is done.)
5. **Monitor**: `$P run remake slurm-status remakefile_regrid.py`, `remake info`, and per-task progress
   (`Completed: X/39` for regrid, `X/N` chunks for coarsen) in
   `/work/scratch-nopw2/mmuetz/um2hp/logs/prod/v7/glm.n2560_RAL3p3_tuned_p4k/`. While this session is open,
   Claude monitors the arrays (state changes, errors, progress).

## Failure handling

Dependency structure: `create_stores` →(afterok)→ `regrid` array →(afterany)→ coarsen trigger → per dim
`coarsen_<dim>_z9` →(afterok)→ `z8` → … → `z0`. Two remake constraints shape recovery:

- A task **cannot be resubmitted while its array is still queued**: `remake run` skips a rule whose array is
  queued, and `remake resubmit` re-executes the whole `submit.sh` (and refuses while any of it is queued).
  Recovery is always: wait for the array to leave the queue, then **replan** with `remake run`, which submits
  only failed/pending tasks (completed ones are skipped) plus a fresh downstream chain.
- Cancelling or failing one element of a coarsen array **blocks all lower zooms of that dim**
  (afterok → DependencyNeverSatisfied). Regrid failures do not block anything: the trigger is afterany and the
  coarsen gating leaves out every batch that needs a missing date (including the whole-period z3–z0 batches).

| Failure | Effect | Action |
|---|---|---|
| create_stores fails | regrid array and trigger never start (DependencyNeverSatisfied) | Read `create_stores.log`; fix; `scancel` regrid array + trigger; rerun launch steps 3–4. If it failed on "Refusing to recreate existing zarr stores", something already wrote stores — investigate before deleting anything. |
| Some regrid tasks fail (OOM at ~91/100G, timeout, S3 error, bad node) | Others continue; trigger still coarsens everything fully regridded | After the regrid array leaves the queue: `remake info -F`; fix (e.g. bump `mem`, add node to `exclude`); `remake run -E slurm remakefile_regrid.py` (only failed tasks); when those succeed, `remake run -E slurm remakefile_coarsen.py` for the batches that were left out. |
| Slow regrid task / node (3–5× seen: host1114, host1240) | Occupies one of 60 slots; only matters if it becomes the last straggler | **Don't cancel mid-array.** Add the node to `exclude` (affects later submissions). If a task is still running ≫ the rest (e.g. >1 h after all others finished), `scancel` it and recover as for a failed regrid task. |
| A coarsen task fails | Lower zooms of that dim blocked | After that rule's array leaves the queue: `scancel` the blocked downstream arrays; fix; `remake run -E slurm remakefile_coarsen.py` (resubmits failed task + downstream chain; completed batches skipped). |
| Slow coarsen task | Holds up the next zoom of that dim | Leave it unless clearly hung (no new `Completed:` lines for a long time) — cancelling means rebuilding the chain as above. |
| Home quota | Tasks fail to record results (seen 2026-09-11) | Keep headroom; the run adds ~250 MB. |
| Input purge | Inputs are on another user's scratch (`scratch-pw6/cscullio`) | Failed input reads show as task failures; recover as above once inputs are restored. |

All recoveries and slow-node cancellations are reported via ntfy (`hk26-jasmin-updates`).

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

## Throughput experiment (2026-09-12)

Regrid throughput collapsed overnight (30/h → 11–15/h) because 8–12 tasks were packed per 1.5 TB node, each
peaking at ~95 GB of its 100 GB limit: the `model_level_to_pressure` step then spends its time in memory
reclaim (one task: 312 min of its 5 h in gaps > 5 min, ~200 min of that in relevel). S3 write retries were
*not* the cause: 53 retry events (5–15 s sleeps) across 382 task logs. 27 tasks hit the 10 h walltime or OOM.

| Setting | Concurrency | Mem/task | Completed sample | Mean duration | Packing |
|---|---|---|---|---|---|
| Launch (18:18–07:57) | 60 | 100G | 336 | 45 min early, 4–10 h once packed | up to 12/node |
| Change 1 (07:57) | 30 | 128G | 19 | **27.9 min** (median 27, 24–38) | max 6/node |
| Change 2 (11:32) | 45 | 128G | (see below) | | |

Applied to the running array with `scontrol update JobId=<id> ArrayTaskThrottle=<n> MinMemoryNode=131072`
(MB, not "128G"); affects queued tasks only. qos=high caps concurrency at 82 tasks at 128 GB.

## Cost of Claude monitoring this run (measured 2026-09-13)

From the session transcript (1,079 assistant messages, 2026-09-11 09:41 → 2026-09-13 13:40):

| | Total | 09-11 (build + launch) | 09-12 (monitor) | 09-13 (monitor + recovery) |
|---|---|---|---|---|
| Output | 890,592 | 701,202 | 103,669 | 85,721 |
| Cache creation | 10,905,539 | 1,927,324 | 4,112,736 | 4,865,479 |
| Cache read | 289,853,122 | 177,858,416 | 57,439,106 | 54,555,600 |
| Fresh input | 2,224 | 1,582 | 354 | 288 |

~11.8 M non-cached tokens (output + cache creation), 290 M cache reads.

**Idle monitoring is not free.** Output fell 8x after the build day, but cache creation *rose* (1.9 → 4.1 → 4.9 M/day):
every monitor event re-primes a growing conversation context, whether or not it needs action. Most events needed
none: packed-node SLOW warnings fired ~60 times and were always benign.

Implications for the next long run:
- Make the watcher filter, not the model: a suppressed event costs nothing, a forwarded one re-primes context.
  `--no-slow`, STALLED (log mtime) instead of SLOW (elapsed), `--interval 300`, `--heartbeat 300` cut event volume
  ~5x on 2026-09-12 with no loss of signal - every real incident (OOM, TIMEOUT, S3 outage) still surfaced.
- Prefer one aggregated PROGRESS line over per-task events; report state *changes*, not states.
- Start with the quiet settings: thresholds were retuned 4 times on the first night (75 → 150 → 240 min stall,
  SLOW 100 → 150 → 240 → 480 min → off), each retune costing a watcher restart and a burst of re-reported events.
- Long-lived monitoring is cheapest attached to a session that is *also* doing work; a purely idle watch still
  pays ~4-5 M cache-creation tokens/day at 1-minute polling.
