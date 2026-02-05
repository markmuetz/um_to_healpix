# Handover for `glm.n1280_CoMA9` processing

* Mark Muetzelfeldt, 5/2/26

Assumes you've already read `installation_and_running.md`. Note the updated command-line script names, e.g. `um-slurm-control`. You will need access to the `hackathon-o` object store tenancy. Once you have access, we should check that you can write to this tenancy using the `s3cmd` tool.

## Setup

1. Go to your `um_to_healpix` top-level dir.
2. Stash any changes you've made in e.g. a new branch, then checkout `main`
3. Get new code: `git pull` 
  (branch `hk26` has been merged into `main`, see this PR: https://github.com/markmuetz/um_to_healpix/pull/1)
4. Activate the conda env `um2hp_env`, then run `pip install -e .` (this will enable the new command-line scripts)
5. DO NOT MODIFY CONFIG. I have given the directories under `/gws/nopw/j04/hrcm/mmuetz/slurm_done/` group write perms, so anyone with access to this GWS should be able to write there (good idea to test first). This means that all jobs which have already run do not need to be rerun, and the weights files for doing to lat/lon to healpix regridding already exist.

## Running

The basic command here is `um-slurm-control -C <config> <subcommand> <args>`, e.g.

* `um-slurm-control -C config/hk26_config.py ls`
* Note this uses the `config/hk26_config.py` file - you should not need to modify this at all.
* You should be able to reproduce the following output:

```shell
(um2hp_env) mmuetz@host839 ~/deploy/um_to_healpix $ um-slurm-control -C config/hk26_config.py ls
glm.n2560_RAL3p3.tuned
glm.n1280_CoMA9
```

* Now check that there are some processing jobs to run. The `--dry-run` option means these will not be submitted, and the `--endtime` option means it will only process up until the start of March. These jobs **have not been done**, i.e. there are no corresponding files at: `/gws/nopw/j04/hrcm/mmuetz/slurm_done/dev/glm.n1280_CoMA9/v6.1` (have a look here to see what has been done).

```shell
(um2hp_env) mmuetz@host839 ~/deploy/um_to_healpix $ um-slurm-control --dry-run -C config/hk26_config.py process --endtime "2020-02-01 00:00" glm.n1280_CoMA9
2026-02-05 10:43:11.452 | WARNING  | um_to_healpix.um_slurm_control:cli:165 - Dry run: not launching any jobs
2026-02-05 10:43:11.453 | INFO     | um_to_healpix.um_slurm_control:process:193 - Running for glm.n1280_CoMA9
2026-02-05 10:43:11.882 | INFO     | um_to_healpix.um_slurm_control:process:245 - 2020-01-23 12:00:00: processing
2026-02-05 10:43:11.883 | INFO     | um_to_healpix.um_slurm_control:process:245 - 2020-01-24 00:00:00: processing
...
2026-02-05 10:43:11.890 | INFO     | um_to_healpix.um_slurm_control:process:245 - 2020-02-01 00:00:00: processing
2026-02-05 10:43:11.890 | WARNING  | um_to_healpix.um_slurm_control:process:212 - Limiting date range to 2020-02-01 00:00:00
2026-02-05 10:43:11.890 | INFO     | um_to_healpix.um_slurm_control:process:260 - Running 18 tasks
2026-02-05 10:43:11.898 | WARNING  | um_to_healpix.um_slurm_control:cli_exit:182 - Dry run: not launching any jobs
```

* Submit some jobs! Remove the `dry-run` option and rerun.
  * For `glm.n1280_CoMA9`, jobs should take 20-40 mins to run, although there is a long tail with some jobs potentially taking hours.
* Once you have successfully processed the entire simulation (i.e. removed the `--endtime` option and all jobs have successfully completed), you can coarsen the data:
* `um-slurm-control -C config/hk26_config.py coarsen glm.n1280_CoMA9`
* All the scripts can be rerun safely - they will only rerun failed/not completed jobs.
    * **Important** only do this if there are no jobs currently running.

## Explainer

What does this do? 
* `um-slurm-control` scans the simulations input directory and works out what jobs can be run, it then scans the donefiles directory and sees what's already been done. 
* It then writes some tasks (simple json files) to `slurm/tasks`. This is an array which will be indexed by the SLURM job array index.
* And a SLURM submit script(s) to `slurm/scripts`
  * This script is a filled in template of `um_slurm_control.py:SLURM_SCRIPT_ARRAY`
  * The script will call `um-process-tasks slurm {tasks_path} ${{ARRAY_INDEX}}` to do the processing
  * Output is written to `slurm/output` - good to check here if there are errors
* Then it `sbatch <script_name>` to queue the jobs

## Monitoring jobs

* `squeue` should show you the jobs pending, then running, like so (you can set up a `watch` on this):

```shell
(base) mmuetz@host839 ~ $ squeue --me --format="%20i %12j %7u %3t %9M %9L %10r %N %80k"
JOBID                NAME         USER    ST  TIME      TIME_LEFT REASON     NODELIST COMMENT                                                                         
65308992_[0-17%40]   regrid       mmuetz  PD  0:00      1-00:00:0 None        glm.n1280_CoMA9,regrid                                                          
(base) mmuetz@host839 ~ $ squeue --me --format="%20i %12j %7u %3t %9M %9L %10r %N %80k"
JOBID                NAME         USER    ST  TIME      TIME_LEFT REASON     NODELIST COMMENT                                                                         
65308992_0           regrid       mmuetz  R   0:01      23:59:59  None       host1118 glm.n1280_CoMA9,regrid                                                          
65308992_1           regrid       mmuetz  R   0:01      23:59:59  None       host1118 glm.n1280_CoMA9,regrid                                                          
...
65308992_17          regrid       mmuetz  R   0:01      23:59:59  None       host1140 glm.n1280_CoMA9,regrid                               
```

* You can see running and completed jobs with `sacct`:

```shell
(base) mmuetz@host839 ~ $ sacct -S now-1hour -o 'jobid%20,start,end,elapsed,state,MaxRSS,NodeList,jobname' -X
               JobID               Start                 End    Elapsed      State     MaxRSS        NodeList    JobName 
-------------------- ------------------- ------------------- ---------- ---------- ---------- --------------- ---------- 
          65308992_0 2026-02-05T10:53:47             Unknown   00:01:54    RUNNING                   host1118     regrid 
          65308992_1 2026-02-05T10:53:47             Unknown   00:01:54    RUNNING                   host1118     regrid 
...
         65308992_17 2026-02-05T10:53:47             Unknown   00:01:54    RUNNING                   host1140     regrid 
```

* On successful completion of the job, job logs will be written to the donedir (`/gws/nopw/j04/hrcm/mmuetz/slurm_done/dev/glm.n1280_CoMA9/v6.1`)
* If there are failed jobs, check in `slurm/output` to see what the errors are. Common ones are: 
  * not being able to access the `kscale` GWS (there is a quick check of this in the SLURM script)
  * timeouts when writing to obj store (this is handled in code and is pretty robust, but you might see)

## Making changes

* If you want to make changes, either modify or copy the config file, then make sure you change the `deploy` variable.
* This will do two things. First, it will use a new donedir, so all jobs will need to be rerun. Second, it will write to a new URL in the object store.
* You can also change the `output_vn` variable, which will have exactly the same effect.