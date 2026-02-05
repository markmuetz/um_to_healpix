import subprocess as sp
from pathlib import Path

import click
import pandas as pd

from um_to_healpix.util import sysrun

SLURM_SCRIPT = """#!/bin/bash
#SBATCH --job-name=jNodeHlth
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --time=00:02:00   
#SBATCH --ntasks=1
#SBATCH --mem=100M
#SBATCH -o slurm/output/node_health_check_%A_%a.out
#SBATCH -e slurm/output/node_health_check_%A_%a.err

target="{target}"

# Run ls with a 5-second timeout to catch the hang
timeout 60s ls $target > /dev/null 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "ERROR:$(hostname):I/O HANG detected on (Timeout)"
    exit 124
elif [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR:$(hostname): Directory missing or inaccessible (Code $EXIT_CODE)"
    exit 99
else
    echo "SUCCESS::$(hostname):healthy."
    exit 0
fi
"""

SLURM_CONFIG = dict(
    account='hrcm',
    partition='standard',
    qos='short',
)

def write_tasks_slurm_job_array(**kwargs):
    now = pd.Timestamp.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")
    slurm_script_path = Path(f'slurm/scripts/script_jasmin_health_check_{date_string}.sh')
    script_kwargs = dict(
        date_string=date_string,
        **SLURM_CONFIG,
    )
    script_kwargs.update(kwargs)

    slurm_script_path.write_text(SLURM_SCRIPT.format(**script_kwargs))
    return slurm_script_path


def sbatch(hostname, slurm_script_path):
    """Submit script using sbatch."""
    try:
        cmd = f'sbatch --deadline=now+10minutes --nodelist={hostname} --comment={hostname} {slurm_script_path}'
        print(cmd)
        return sysrun(cmd).stdout.strip()
    except sp.CalledProcessError as e:
        print(f'sbatch failed with exit code {e.returncode}')
        print(e)
        raise


@click.group()
def cli():
    pass

@cli.command
@click.argument('target')
@click.option('--hostnames', default=None)
def target(target, hostnames):
    if hostnames:
        hostnames = hostnames.split(',')
    else:
        cmd = 'sinfo -p standard -t idle,mix,alloc,comp -h -o "%N"'
        result = sp.run(cmd, capture_output=True, text=True, shell=True)
        lines = [l for l in result.stdout.split('\n') if l]
        line = lines[0]
        # Parse out hostnames from ranges in form:
        # host[1004-1043,1045-1116,1118-1132,1134-1135,1137-1170,1183-1272]
        hostnums = line[5:-1]
        hostranges = hostnums.split(',')
        hostids = []
        for hostrange in hostranges:
            if '-' in hostrange:
                start, end = hostrange.split('-')
            else:
                start = end = hostrange
            hostids.extend(list(range(int(start), int(end) + 1)))
        hostnames = [f'host{i}' for i in hostids]

    slurm_script_path = write_tasks_slurm_job_array(target=target)

    jobids = []
    for hostname in hostnames[:5]:
        jobids.append(sbatch(hostname, slurm_script_path))
    # print(jobids)
    # jobids_range = f'{jobids[0]}..{jobids[-1]}'
    print("sacct -P -o 'jobid%20,start,end,elapsed,state,MaxRSS,NodeList' --name=jNodeHlth|grep -E '^[0-9_]*\.batch\|'")

@cli.command
def sacct():
    cmd = r"sacct -P -o 'jobid%20,start,end,elapsed,state,MaxRSS,NodeList' --name=jNodeHlth|grep -E '^[0-9_]*\.batch\|' --color=Never"
    result = sp.run(cmd, capture_output=True, text=True, shell=True)
    lines = [l for l in result.stdout.split('\n') if l]
    data = [line.split('|') for line in lines]
    df = pd.DataFrame(data, columns=["jobid", "start", "end", "elapsed", "state", "maxrss", "host"])
    print(df)
    breakpoint()


if __name__ == '__main__':
    cli()