"""Parse and summarize SLURM task status/resource usage for a job family.

This module wraps ``sacct``/``squeue`` calls to collect status, timing, and
memory information for batch tasks that belong to a given parent job ID (or ID
prefix), then prints compact progress statistics useful during production runs.

It also includes optional interactive helpers for opening log files and ranking
hosts by historical task runtime.
"""

from pathlib import Path
import subprocess as sp

import click
import pandas as pd


def run_parse_cmd(jobids_str):
    """Fetch SLURM accounting/queue info and return it as a normalized DataFrame.

    Parameters
    ----------
    jobids_str : str
        Job id or comma-separated job ids passed to ``sacct -j`` and used as a
        prefix filter for pending tasks from ``squeue``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing task-level records with columns including
        ``jobid``, ``start``, ``end``, ``elapsed``, ``state``, ``maxrss`` (GB),
        and ``host``. Pending tasks have only ``jobid`` and ``state``.
    """
    # Gets a list of all running/completed tasks for JOB ID, selecting just the ones that end with .batch (these
    # are the ones that give the correct memory info) using grep (this effectively removes the header), separated by |
    # (-P).
    # Query accounting data in parseable '|' format and keep only .batch records.
    cmd = rf"sacct -P -o 'jobid%20,start,end,elapsed,state,MaxRSS,NodeList' -j {jobids_str}|grep -E '^[0-9_]*\.batch\|' --color=Never"
    result = sp.run(cmd, capture_output=True, text=True, shell=True)

    # Parse command output into structured rows/columns.
    lines = [l for l in result.stdout.split('\n') if l]
    data = [line.split('|') for line in lines]
    df = pd.DataFrame(data, columns=["jobid", "start", "end", "elapsed", "state", "maxrss", "host"])

    # Convert timestamps; invalid/missing values become NaT.
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df['end'] = pd.to_datetime(df['end'], errors='coerce')

    # Convert MaxRSS from SLURM suffix format (e.g. "1536M") to GB float.
    df['maxrss'] = df['maxrss'].map(map_maxrss) / 1e9
    # Parse elapsed duration strings to Timedelta.
    df["elapsed"] = pd.to_timedelta(df["elapsed"])

    # Collect info on pending tasks.
    # returns a list of all pending task IDs, no header (-h) and expanding batch tasks (-r) for me.
    # Query queue for PENDING tasks owned by current user.
    cmd2 = rf'squeue --me --format="%20i" --state=PD -r -h'
    result2 = sp.run(cmd2, capture_output=True, text=True, shell=True)
    lines2 = [l for l in result2.stdout.split('\n') if l]

    # Keep pending tasks related to the requested job id prefix, and normalize
    # to match sacct's '.batch' jobid convention.
    data2 = [line.strip() for line in lines2]
    data2 = [[v + '.batch', 'PENDING'] for v in data2 if v.startswith(jobids_str)]
    df2 = pd.DataFrame(data2, columns=["jobid", "state"])

    # Merge accounting rows with queue-only pending rows.
    df = pd.concat([df, df2], ignore_index=True)

    return df


def map_maxrss(value):
    """Convert SLURM MaxRSS suffix string into bytes.

    Parameters
    ----------
    value : str | int | float
        SLURM MaxRSS-style value such as ``'1024K'``, ``'250M'``, ``'1.5G'``.

    Returns
    -------
    float
        Value in bytes. Empty string maps to ``0``.
    """
    # Normalize input to string to simplify downstream parsing.
    value = str(value)
    if value == '':
        return 0
    # Binary prefixes used by SLURM memory reporting.
    suffix_factor = {
        'K': 2**10,
        'M': 2**20,
        'G': 2**30,
        'T': 2**40,
    }
    # Last char is suffix, remaining prefix is numeric portion.
    return float(value[:-1]) * suffix_factor[value[-1]]


def print_df(df, full):
    """Print status counts and summary statistics for parsed SLURM records.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from ``run_parse_cmd``.
    full : bool
        If True, print full non-pending task table before summary.
    """
    # Completed subset is used for duration/resource statistics.
    df_comp = df[df.state == 'COMPLETED']
    # Exclude pending rows from optional full table to avoid sparse columns.
    df_print = df[df.state != 'PENDING']

    if full:
        print(df_print.to_string())
        print()

    # Print task counts per major SLURM state (if present).
    state_value_counts = df.state.value_counts()
    for state in ['PENDING', 'RUNNING', 'COMPLETED', 'CANCELLED', 'FAILED']:
        if state in state_value_counts:
            value = state_value_counts[state]
            print(f"{state:<10}: {value}")

    # Always show total rows tracked.
    total = len(df)
    print(f"TOTAL     : {total}")

    # Wall-clock bounds from completed tasks.
    print()
    print('earliest start:', df_comp.start.min())
    print('latest end    :', df_comp.end.max())
    print('total duration:', df_comp.end.max() - df_comp.start.min())

    # Summary stats for completed elapsed duration and maxrss.
    print()
    varname = 'elapsed'
    print(f'Completed {varname}')
    for meth in ['min', 'mean', 'max']:
        print(f'  {meth:<5}: {getattr(df_comp[varname], meth)()}')
    varname = 'maxrss'
    print(f'Completed {varname}')
    for meth in ['min', 'mean', 'max']:
        print(f'  {meth:<5}: {getattr(df_comp[varname], meth)():.1f}G')

    # Simple throughput-based ETA:
    # expected remaining = (mean completed duration) * (tasks to go / running).
    ntogo = len(df[df.state.isin(['PENDING', 'RUNNING'])])
    nrunning = len(df[df.state == 'RUNNING'])
    ncompleted = len(df[df.state == 'COMPLETED'])
    if nrunning and ncompleted:
        est_finish_duration = df_comp['elapsed'].mean() * ntogo / nrunning
        print()
        print(f'Est. finished duration: {est_finish_duration}')
        print(f'Est. finished time: {pd.Timestamp.now() + est_finish_duration}')


# Useful in interactive mode.
def view_logs(df):
    """Open matching SLURM error logs in Vim for selected jobs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a ``jobid`` column with ``.batch`` suffix.
    """
    # Convert each '<job>.batch' to a glob for corresponding slurm error files.
    jobids = [list(Path.cwd().glob('slurm/output/*' + j[:-6] + '*.err'))[0] for j in df.jobid.values.tolist()]
    # Launch a single Vim session with all matching log paths.
    sp.run(['vim'] + jobids)


# Useful in interactive mode.
def fastest_hosts(df, min_num_runs=3, nhosts=40):
    """Print fastest hosts by mean elapsed time from completed tasks.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``state``, ``host``, and ``elapsed`` columns.
    min_num_runs : int, default=3
        Minimum number of completed tasks per host to include in ranking.
    nhosts : int, default=40
        Maximum number of top hosts to display.
    """
    # Use only completed tasks for fair runtime comparisons.
    df_comp = df[df.state == 'COMPLETED']
    # Aggregate host-level runtime mean and sample count.
    host_stats = df_comp.groupby('host')['elapsed'].agg(['mean', 'count'])
    # Filter out hosts with insufficient sample size.
    host_stats = host_stats[host_stats['count'] >= min_num_runs]
    # Smallest mean elapsed => fastest hosts.
    top_40 = host_stats.sort_values(by='mean').head(nhosts)
    print(top_40.to_string())

    # Emit convenient SBATCH prefer list for reuse.
    node_string = ",".join(sorted(top_40.index.tolist()))
    print(f"#SBATCH --prefer={node_string}")


@click.command
@click.option('--full', '-F', is_flag=True)
@click.option('--interactive', '-I', is_flag=True)
@click.argument('jobids_str')
def main(full, interactive, jobids_str):
    """CLI entrypoint for SLURM parsing and summary output.

    Parameters
    ----------
    full : bool
        Print full non-pending task table before summary.
    interactive : bool
        If True, drop into debugger after printing summary.
    jobids_str : str
        Job id expression passed to underlying parser command.
    """
    # Collect state/resource information from SLURM tools.
    df = run_parse_cmd(jobids_str)
    # Print compact progress + performance summary.
    print_df(df, full)
    if interactive:
        # Handy for ad-hoc exploration (e.g., call view_logs/fastest_hosts).
        breakpoint()