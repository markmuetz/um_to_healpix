# Syncs the repo to the JASMIN transfer node. Can be run from anywhere inside the repo.
# cd to repo root first so --filter=':- .gitignore' correctly reads .gitignore,
# then rsync from . so the filter applies to all subdirectories.
# Run with: bash deploy/rsync_to_jasmin.sh
cd "$(git rev-parse --show-toplevel)"
rsync -vzar --filter=':- .gitignore' --exclude=.git --exclude=*.pyc . mmuetz@xfer-vm-01.jasmin.ac.uk:/home/users/mmuetz/deploy/um_to_healpix
