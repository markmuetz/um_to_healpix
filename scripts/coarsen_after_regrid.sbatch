#!/bin/bash
# Plan and submit the coarsen pipeline once a regrid array has finished, unattended:
#     sbatch --dependency=afterany:<regrid array job id> scripts/coarsen_after_regrid.sbatch
# afterany: coarsen whatever is fully regridded even if some regrid tasks failed (the coarsen matrix only
# includes batches whose .pp dates all have a successful regrid task). Rerun (or run
# `remake run remakefile_coarsen.py -E slurm`) after fixing failed regrid tasks to pick up the rest.
#SBATCH --job-name=coarsen_trigger
#SBATCH --account=hrcm
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=/work/scratch-nopw2/mmuetz/um2hp/logs/coarsen_trigger_%j.out

set -euo pipefail
cd /home/users/mmuetz/projects/local/hk26/um_to_healpix
~/.pixi/bin/pixi run remake info remakefile_regrid.py
~/.pixi/bin/pixi run remake run -E slurm remakefile_coarsen.py
