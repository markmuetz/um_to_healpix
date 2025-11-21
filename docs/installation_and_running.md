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

## Test
* This should have installed the `um_to_healpix` package which has a CLI entry point
* `um_to_healpix --help`

# Running
Assumes that you are doing this on JASMIN.

* First, decide which simulation to process.
  * `um_to_healpix ls`
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