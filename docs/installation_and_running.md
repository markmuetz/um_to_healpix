# Installation

The recommended way to install is by cloning the git repo, creating a conda env, then pip installing.

## Clone repo 
* `git clone https://github.com/markmuetz/um_to_healpix`
* `cd um_to_healpix`

## Setup/activate conda env
* `conda env create -f envs/um2hp_env.yml`
* `conda activate um2hp_env`
* Make the conda env usable in https://notebooks.jasmin.ac.uk/
* `python -m ipykernel install --user --name um2hp_env`

## Pip install
* `pip install -e .`

## Test processing command
* This should have installed the `um_slurm_control` package which has a CLI entry point
* `um_slurm_control --help`

## Access to JASMIN GWSs
* You will need access to the `kscale` and `hrcm` GWSs
  * Find these and apply here: https://accounts.jasmin.ac.uk/services/group_workspaces/

## Access to JASMIN Object Store
* For info on the object store, look here: https://help.jasmin.ac.uk/docs/short-term-project-storage/using-the-jasmin-object-store/
* You will need access to the `hackathon-o` tenancy in the JASMIN object store
  * Find this and apply here: https://accounts.jasmin.ac.uk/services/object_store/
* Once you have access, you need to create an access key:
  * (See here: https://help.jasmin.ac.uk/docs/short-term-project-storage/using-the-jasmin-object-store/#using-s3cmd)
  * Go to https://s3-portal.jasmin.ac.uk/object-store
  * Click "Manage Object Store"
  * Click "Create Key" tab
  * Add a description like "wcrp_hackathon"
  * **Take note of the S3 Secret Key -- you will not be able to see it again!**
  * Use this info to make a file in your home dir on JASMIN `.s3cfg` (filling in the blanks):
```
access_key = <access-key>
host_base = hackathon-o.s3.jc.rl.ac.uk
host_bucket = hackathon-o.s3.jc.rl.ac.uk
secret_key = <secret-key>
use_https = False
signature_v2 = False
```

## Test access to JASMIN Object Store using `s3cmd`
* With `~/.s3cfg` set up, on JASMIN you *should* be able to:
  * `s3cmd ls s3://sim-data/`
  * `s3cmd ls s3://sim-data/dev/v5.2/glm.n2560_RAL3p3/um.PT1H.hp_z10.zarr/`
* And see nice output. See here for more info on `s3cmd`: https://help.jasmin.ac.uk/docs/short-term-project-storage/using-the-jasmin-object-store/#using-s3cmd

# Running processing
Assumes that you are doing this on JASMIN, and that you have access to the `kscale` and `hrcm` (high-resolution climate modelling) GWSs, with access to the `hackathon-o` object store tenancy set up as above.

* First, decide which simulation to process.
  * `um_slurm_control ls`
  * Currently
    * `glm.n1280_CoMA9
glm.n2560_RAL3p3
glm.n1280_GAL9_nest
SAmer_km4p4_RAL3P3.n1280_GAL9_nest
Africa_km4p4_RAL3P3.n1280_GAL9_nest
SEA_km4p4_RAL3P3.n1280_GAL9_nest
SAmer_km4p4_CoMA9_TBv1.n1280_GAL9_nest
Africa_km4p4_CoMA9_TBv1.n1280_GAL9_nest
SEA_km4p4_CoMA9_TBv1.n1280_GAL9_nest
CTC_km4p4_RAL3P3.n1280_GAL9_nest
CTC_km4p4_CoMA9_TBv1.n1280_GAL9_nest
`
* Then, make sure you have edited `src/um_to_healpix/um_processing_config.py`
  * Edit `output_vn` and `deploy`
  * Changing either will force new processing (by using a different done file path)
  * It will also determine where the output is put in the JASMIN object store
  * You will need to edit the `weightsdir` and `donedir` to directories where you have write access.
* You can view all config for a simulation:
  * `um_slurm_control print-config glm.n1280_CoMA9`
* Set up a SLURM monitoring script on JASMIN to see progress
  * `watch -n10 squeue -u $USER`
* Run the processing (e.g.):
  * `um_slurm_control -D process glm.n2560_RAL3p3`
  * On first run for unique `output_vn`/`deploy`, this will create an empty zarr store in the JASMIN object store
  * Other runs, the dates which have not been processed will be processed
    * A SLURM array is used, with a limit on the total number of jobs so as not to swamp the object store
    * Initially, this appears as one array job, then expands as individual jobs are launched
  * On completion of a job, appropriate `done` file is rewritten
  * To force a specific job to rerun, just delete the done file
  * To force *all* jobs to rerun, edit config as above
  * Output will be put in the `slurm/output/` directory

# View output

* Go to https://notebooks.jasmin.ac.uk/
* Navigate to the `/um_to_healpix/ctrl/notebooks` directory
* Open the `view_UM_healpix_on_JASMIN_obj_store.ipynb` notebook
* Try running it.