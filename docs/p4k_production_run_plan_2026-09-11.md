# Production run plan: glm.n2560_RAL3p3_tuned_p4k with remake3 (2026-09-11)

**Status: COMPLETE 2026-09-13 15:18** (45 h; launched 2026-09-11 18:18). Verified — see "Outcome" and
"Changes required before the next run" at the end of this document.

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

---

# Outcome (2026-09-13)

`s3://sim-data/prod/v7/glm.n2560_RAL3p3_tuned_p4k/` is complete: 22 stores (PT1H/PT3H × z0–z10),
9,745 hourly and 3,249 3-hourly time steps, 2020-01-20 → 2021-03-01.
remake: `regrid` 813/813 and `coarsen` 1,462/1,462 tasks up to date, 0 failed, 0 pending.

| Stage | Tasks | Wall clock | Notes |
|---|---|---|---|
| create_stores | 1 | 31 min | weights already existed |
| regrid | 812 | 18:49 Fri → 00:36 Sun | 785 first pass + 27 reruns (done 03:01 Sun) |
| coarsen z9 | 1,300 | 03:03 → 09:17 Sun | 1,262 first pass + 38 reruns after the S3 outage |
| coarsen z8–z0 | 162 | 09:17 → 15:18 Sun | 3d chain rerun at 128 GB after OOM |
| **Total** | **2,275** | **45 h** | inside the 48 h target |

## Verification

`scripts/compare_stores.py` vs the control sim (`glm.n2560_RAL3p3.tuned`, prod) at z10, z5, z2, z0:

- **Coverage complete**: every variable has all time steps written, `gaps=0`, full range to 2021-03-01.
  (3-hourly instantaneous fields start 2020-01-20T00, time-means 2020-01-20T03 — expected.)
- **Structure identical** to the control: variables, coords, chunking. Only differing global attribute is
  `simulation_description`.
- **NaN patterns identical** (0 mismatches), i.e. the land/sea and missing-data masks agree.
- **Differences physically sensible for +4K SST**: `tas` +4.87/+4.51/+5.09 K at three dates spread over the
  period, `rlut` +11.5…+12.1 W m-2, `pr` +3.5…+7.1e-06 kg m-2 s-1, `psl` +35…+54 Pa, `clt` −0.9…−2.3 %,
  `uas` +0.15 m s-1.
- The "188 flagged" entries in each report are by construction: with `--b-key` every field differs; the huge
  `max_rel` values are divisions by near-zero values (cloud ice), so `meanDiff` is the meaningful column.

## Incidents

1. **Regrid memory thrashing (the expensive one).** `mem=100G` against a measured peak of 98.5 GB (mean 84.6 GB
   over 812 tasks). When SLURM packed 8–12 tasks onto a 1.5 TB node, the `model_level_to_pressure` step spent its
   time in memory reclaim: one task logged 312 min of gaps > 5 min in a 5 h run, ~200 min of it in relevel.
   Throughput fell from ~30/h to 11–15/h overnight; 23 tasks hit the 10 h walltime, 1 OOMed.
   Fixed mid-run with `scontrol` (128 GB, throttle 45): per-task mean fell to ~45 min and throughput rose to ~49/h.
2. **S3 outages.** 06:14–06:30 and 15:32–15:42 on 2026-09-13. Bucket listings timed out from compute nodes *and*
   from the login node, so not caused by our load (though ~140 concurrent tasks at the time will not have helped).
   Cost: 38 coarsen z9 batches, because reads/listings have no retry logic — writes do
   (`util.async_da_to_zarr_with_retries`, 53 benign retry events across 382 regrid logs).
3. **coarsen 3d z8 OOM.** `mem=32G` for every coarsen rule; 3d z8 peaks at 42.7 GB (each z8 batch reads ~16× a z9
   batch). 18 of 21 tasks were killed; rerun at 128 GB succeeded.
4. **One genuinely slow node**, host1239 (~20× slower, not load-related) — excluded. All other "slow nodes" were
   just busy: the eight fastest and slowest nodes are identical hardware (AMD EPYC 9654, 192 cores, 1.54 TB).

## Measurements

**Regrid** (n=812): MaxRSS mean 84.6 GB, max 98.5 GB. CPU: `nproc=6` hardcoded (`um_process_tasks.py:88`),
matching `cpus-per-task=6`.

Throttle experiment at 128 GB (mean task duration / completions per hour):

| Throttle | Mean duration | Completions/h |
|---|---|---|
| 30 | 53.7 min | 22 |
| **45** | **51–59 min** | **48–49** |
| 60 | 66.3 min | 39–41 |

**Coarsen** (peak RSS and median duration per rule, 32 GB requested except 3d ≥ z8 which used 128 GB):

| Rule | Tasks | Peak RSS | Median | Rule | Tasks | Peak RSS | Median |
|---|---|---|---|---|---|---|---|
| 2d z9 | 975 | 2.6 GB | 28 min | 3d z9 | 325 | 12.0 GB | 36 min |
| 2d z8 | 61 | 6.0 GB | 43 min | 3d z8 | 21 | **42.7 GB** | 104 min |
| 2d z7 | 16 | 4.7 GB | 19 min | 3d z7 | 6 | **42.8 GB** | 33 min |
| 2d z6 | 16 | 1.9 GB | 8 min | 3d z6 | 6 | 10.7 GB | 9 min |
| 2d z5 | 16 | 0.8 GB | 6 min | 3d z5 | 6 | 2.7 GB | 4 min |
| 2d z4–z0 | 8 | ≤0.8 GB | 1–5 min | 3d z4–z0 | 6 | ≤1.9 GB | 0–3 min |

**Coarsen CPU: 0.11 cores (2d) and 0.30 cores (3d) of the 12 requested** — these tasks are I/O bound on S3.
Aggregate throughput was ~215 batches/h whether 37 or 111 tasks ran concurrently, i.e. S3-limited, so extra
concurrency only divides the same bandwidth (and the 06:14 failures came while concurrency was highest).

**Node packing vs speed** (regrid, tasks started after 12:00, 35 nodes, identical hardware):

| Our tasks on the node | Nodes | Mean duration |
|---|---|---|
| 3–4 | 11 | 60 min |
| 5–7 | 15 | 74 min |
| 8+ | 7 | 104 min |

Correlation is only 0.04 because *other users'* load matters as much: host1222 had 3 of our tasks and a 170 min
mean at CPU load 175/192; host1030 had CPU load 0 and a 27 min mean. The node's ratio is 8 GB/core; a regrid task
wants 95 GB with 6 cores (16 GB/core), so SLURM fills a node's memory long before its cores.

---

# Changes required before the next run

Ordered by value. Items 1–3 are enough to avoid every incident above.

## 1. Right-size the resource requests (remakefiles)

`remakefile_regrid.py`, `regrid` rule: `mem` 100G → **128G**, `array_throttle` 60 → **45**.

`remakefile_coarsen.py`, per-rule instead of a flat 32 GB + 12 CPUs. Peak + ~50 % for a first run, then peak + 25 %:

| Rule | Peak seen | Set |
|---|---|---|
| 2d z9 | 2.6 GB | 8 GB |
| 2d z8–z0 | ≤6.0 GB | 16 GB |
| 3d z9 | 12.0 GB | 24 GB |
| 3d z8, z7 | 42.8 GB | 64 GB |
| 3d z6–z0 | ≤10.7 GB | 24 GB |

`cpus-per-task` 12 → **2** for all coarsen rules (measured 0.11–0.30 cores).

**N.B. editing a rule changes its code hash, so all completed tasks go stale.** After editing, re-stamp:
`remake set-state remakefile_<x>.py -Q '<succeeded tasks>' --success` — otherwise the next `remake run`
resubmits all 2,275 tasks (and `create_stores` would refuse, since the stores now exist).

## 2. Retry S3 reads and listings

Reads and `open_zarr`/listing calls have no retry logic; a momentary failure kills the task. Wrap them as writes
already are (`util.async_da_to_zarr_with_retries`), and/or set `botocore.config.Config(read_timeout=…,
retries={'max_attempts': 10, 'mode': 'adaptive'})` in `get_jasmin_s3()`. This alone would have saved the 38
batches lost on 2026-09-13, and would make the pipeline robust to the ~10–16 min outages seen twice in one day.

## 3. Cut regrid's peak memory (~95 GB) - DONE 2026-09-14

Implemented (commit 8957a2d): model-level variables are interpolated and regridded one time step at a time, in
float32, so the (12, 25, 3841, 5120) float64 array (44 GB) never exists.

| | before | after |
|---|---|---|
| peak RSS, whole task | 98.5 GB | **60.1 GB** |
| duration (quiet node) | 27-53 min | 27 min |
| memory request | 128 GB | **96 GB** (~1.6x peak; caps SLURM at 16 of our tasks per 1.5 TB node) |

Validated against the prod p4k store for 2020-03-01: `tas`, `pr` and `ta` bit-identical; `cli` and `clw` differ
by at most 2.1e-07 relative (float32 interpolation, agreed with the user), NaN patterns identical.

What is left at 60 GB is mostly the *pressure-level* 3d variables, which are read whole from the .pp files
(12 times x 25 levels x 19.7M points = 24 GB in float32) and are untouched by this change. Processing those per
time step too would take the peak to ~20 GB, at the cost of more, smaller reads.

**The memory request is also the packing lever.** Regrid uses 1.15 of its 6 CPUs (max 1.85), so CPU never limits
how many tasks SLURM puts on a node - only memory does. Requesting close to the true peak would allow 24-32 of
our tasks per node, and density is what drove duration in the p4k run (3-4/node 60 min; 8+/node 104 min). 96 GB
is the compromise: enough headroom over the 60 GB peak, and a 16/node cap. Requesting less would need another
way to control density.

## (original notes)

### Cutting regrid's peak memory

The root cause of incident 1: at 16 GB/core the task cannot be scheduled without oversubscribing node memory.
Process one group at a time and free cubes, and/or chunk the vertical interpolation instead of holding all model
levels for all variables. Halving it to ~45 GB would match the nodes' 8 GB/core ratio and remove the thrashing
risk; it also doubles how many tasks fit per node.

## 4. Wire `nproc` to the CPU request

`um_process_tasks.py:88` hardcodes `nproc=6`. Read `SLURM_CPUS_PER_TASK` (as `um_process_tasks.py:590` already
does) so that asking for more CPUs actually speeds up `_regrid_easygems_delaunay_parallel` (34 min of gaps in the
slow task analysed) rather than idling, and so CPUs can be requested in proportion to memory.

## 5. Verification and sanity plots as remake rules

Agreed design (2026-09-13), to run **after** the whole coarsen chain:

- `check_store`, matrix `(config_key, freq, zoom)` (22 tasks), `depends_on=[coarsen_2d_z0, coarsen_3d_z0]`, output
  one JSON report per store. Checks: coverage (all chunks written, no gaps, full time range), NaN patterns,
  per-variable ranges (reuse `output_tests/datasets.py: RANGE_CHECKS`), and level-to-level consistency
  (a coarsened field ≈ the mean of its parents). Reference comparison against a control sim stays optional
  (`--b-key`), since it only makes sense for perturbation pairs.
- Sanity plot rules depending on `check_store`, so plots only exist for verified data. Reuse
  `um_to_healpix.plotting`: `plot_zonal_mean` (pr across zooms), `plot_all_fields`, `plot_timeseries`,
  and the clw pressure profile — `output_tests/test_plots.py` already has all four as pytest cases.
- Select over **any** extra dimension, not just `pressure`: `mrsol` is hourly but has `depth` (the `2d_depth`
  group), which is what broke `compare_stores.py` twice this weekend. Drive this off the variable's dims, not the
  config group name, and apply range checks per level.
- Keep `scripts/compare_stores.py` as the library the rules call, and as the ad-hoc tool.

## 6. Smaller things

- `compare_stores.py`: two bugs found and fixed this weekend — coverage counted zarr *chunk* indices as time
  indices (wrong by the time-chunk factor below z9, commit 2b5c0ce), and the max-difference cell lookup broke on
  variables with an extra dimension (84537bc). Consider a `--coverage-only` mode for cheap re-checks.
- Node exclusion is rarely worth it: only host1239 was genuinely faulty. Today's busy node is tomorrow's fast one.
- `--exclusive=user` would insulate tasks from other users' load, at the cost of idle cores; worth testing if
  regrid's memory footprint cannot be reduced.
- Housekeeping: the `dev_remake` test stores for the tuned sim (22 stores, Jan 20–31 2020) are still on S3 and can
  be deleted once nobody needs the comparison.
- The `.pp` source for p4k (`/work/scratch-pw6/cscullio/.../n2560_RAL3p3_tuned_p4k`, **43 TB**, owned by
  cscullio) is **free to delete**: processing is complete and verified, and the zarr stores hold every variable
  in the protocol we were given. Recorded for the future, since the streams contain more than the protocol asks
  for: the pipeline reads `apvera`–`apverd` and ignores `apvere`, which carries 14 STASH codes found in no other
  stream (`m01s01i201`, `m01s03i332`, `m01s02i204`, `m01s04i209`, `m01s09i218`, and `*i517`–`*i520` in sections
  01 and 02). Those are out of scope for the protocol, so their loss is expected rather than a gap.

## 7. Conservative regridding via grid-doctor (future)

**Why.** The current regrid is easygems Delaunay barycentric interpolation, which is *not conservative*: domain
means of precipitation and radiative fluxes are not preserved from the N2560 source to z10. Users of this dataset
will compute exactly those integrals. [grid-doctor](https://github.com/freva-org/grid-doctor) (DKRZ/freva, BSD)
wraps ESMF and offers `method="conservative"` as well as `"nearest"`.

**Plan: add it as an option first**, so the two can be compared like for like on the same dates. We are not
looking for bit correspondence — a close match (low RMSE, high correlation) is the acceptance bar, with the
conservative path expected to *differ deliberately* on integral quantities.

The seam is narrow, since both are compute-weights-once/apply-many:

| ours | grid-doctor |
|---|---|
| `gen_weights(da, weights_path, zoom, ...)` | `compute_healpix_weights(ds, level, method=...) -> Path` |
| `LatLon2HealpixRegridder.regrid(da, lonname, latname)` | `apply_weight_file(ds, weights_path, missing_policy=...)` |

A `regrid_method` config key behind `regrid_da_to_healpix()` plus the two `gen_weights` call sites
(`um_process_tasks.py:277`, `:330`) is essentially the whole change.

Feasibility is good: `ESMF_RegridWeightGen`, `mpirun` and `esmpy` 8.9.1 are **already in the pixi env**, so the
offline MPI weight-generation path (better suited to a SLURM job than the in-memory one) needs no new system
dependencies. Only `grid_doctor` and its dep `healpix-geo` are missing.

Things that will bite, in order:

1. `weights_filename()` (`um_process_tasks.py:160`) has no method component, so the two weight sets would collide
   at the same path. This is the prerequisite for any comparison.
2. `missing_policy` (`renormalize` vs `propagate`) changes coastline NaN behaviour, which feeds straight into
   `checks.py`'s `MAX_NAN_FRACTION_BY_VAR` and `ALLOW_EDGE_MISSING`. Without per-method thresholds a methodology
   change reads as a regression.
3. Dim naming: ours emits `healpix_index`, grid-doctor uses `cell` (the same mismatch that hit `attach_coords` in
   the plotting rule). Rename to keep the stores drop-in compatible.
4. `add_cyclic` should be **off** for the grid-doctor path: `_xr_add_cyclic_point` and the `hp_lon[hp_lon==0]=360`
   hack exist only because Delaunay needs the source convex hull to cover the target. ESMF handles periodicity
   natively, so this is a real simplification rather than a porting detail.

**Cheapest first experiment**, using tooling that already exists: regrid one date both ways into the `dev_remake`
stores (the path used for the memory measurement in item 3), then compare with `scripts/compare_stores.py
--b-key`. Roughly 30 minutes of compute, before committing to any refactor. Acceptance is a close match (low
RMSE, high correlation), not bit correspondence. Note the p4k `.pp` source is being deleted, so pick a sim whose
source is still on disk — the tuned control, which is what the `dev_remake` stores already hold.

Whether this should then be applied consistently across *all* datasets is a separate question, and would mean
reprocessing what has already been published.

**Caution:** grid-doctor's own README calls it "a scripting solution for a proof of concept" and it is classified
Alpha, so pin a git rev rather than tracking `main`.

## Operational notes (worked well, keep)

- A running array's settings can be changed without resubmitting: `scontrol update JobId=<id>
  ArrayTaskThrottle=<n>`, `MinMemoryNode=<MB>` (megabytes — "128G" is rejected), `ExcNodeList=<nodes>`.
  Only queued elements are affected, so it is a safe way to experiment mid-run.
- To change resources for a *remake* submission without making tasks stale: `remake run -n -E slurm` to write the
  scripts, edit `.remake/slurm/<rule>.sbatch`, then `remake resubmit`. Used for both recoveries.
- remake constraints to plan around: a rule's tasks cannot be resubmitted while its array is queued (wait, then
  replan); a failed/cancelled task in one zoom blocks every lower zoom of that dim (`afterok`), so recovery is
  "cancel the blocked arrays, then `remake run`", which skips completed tasks and rebuilds the chain.
- The coarsen gating (only batches whose `.pp` dates all have a *successful* regrid task) meant partial regrid
  never produced partial coarsened data, and incremental reruns picked up exactly what was missing.
