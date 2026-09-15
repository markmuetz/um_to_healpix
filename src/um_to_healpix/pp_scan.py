"""Scanning input .pp directories and mapping dates to the files that hold them.

Split out of the old um_slurm_control.py (which submitted SLURM jobs itself; the remake pipelines do that now)
so the scan is usable without dragging in a CLI or a job-submission template.
"""
from collections import defaultdict

import pandas as pd
from loguru import logger

# Original DYAMOND3 layout. The +4K reruns use 'apve*/*.pp' (no field.pp/ level); set `pp_glob` in the config.
DEFAULT_PP_GLOB = 'field.pp/apve*/**/*.pp'
# Streams that make up one complete date. apvere is deliberately ignored: it appears after ~2020-02-20 and
# carries diagnostics outside the processing protocol.
NSTREAMS = 4


def parse_date_from_pp_path(path):
    """Date from a .pp filename, e.g. ...apvera_20200120T0000Z.pp -> 2020-01-20 00:00."""
    datestr = path.stem.split('.')[-1].split('_')[1]
    if datestr[-1] == 'Z':
        return pd.to_datetime(datestr, format='%Y%m%dT%H%MZ')
    return pd.to_datetime(datestr, format='%Y%m%dT%H')


def find_dyamond3_pp_dates_to_paths(basedir, pp_glob=DEFAULT_PP_GLOB):
    """Search for pp_paths with a specific date (N.B. filename sensitive)."""
    pp_paths = sorted(basedir.glob(pp_glob))
    logger.debug(f'found {len(pp_paths)} pp paths')
    pp_paths = [p for p in pp_paths if p.is_file()]
    dates_to_paths = defaultdict(list)
    for path in pp_paths:
        if 'apvere' in path.stem:
            continue
        dates_to_paths[parse_date_from_pp_path(path)].append(path)
    # Only keep completed downloads.
    dates_to_paths = {k: v for k, v in dates_to_paths.items() if len(v) == NSTREAMS}
    logger.debug(f'found {len(dates_to_paths)} complete dates')
    return dates_to_paths


# Backwards-compatible alias (the old name was private).
_parse_date_from_pp_path = parse_date_from_pp_path
