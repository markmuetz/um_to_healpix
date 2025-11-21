# Installation

The recommended way to install is by cloning the git repo, creating a conda env, then pip installing.

## Clone repo 
* `git clone https://github.com/markmuetz/um_to_healpix`
* `cd um_to_healpix`

## Setup/activate conda env
* `conda env create -f envs/um2hp_env.yml`
* `conda activate um2hp_env`

## Pip install
* `pip install -e .`

## Test
* This should have installed the `um_to_healpix` package which has a CLI entry point
* `um_to_healpix --help`