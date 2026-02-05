from pathlib import Path
import subprocess as sp

import click
import pandas as pd


def run_parse_cmd(jobids_str):
    # Gets a list of all running/completed tasks for JOB ID, selecting just the ones that end with .batch (these
    # are the ones that give the correct memory info) using grep (this effectively removes the header), separated by |
    # (-P).
    cmd = rf"sacct -P -o 'jobid%20,start,end,elapsed,state,MaxRSS,NodeList' -j {jobids_str}|grep -E '^[0-9_]*\.batch\|' --color=Never"
    result = sp.run(cmd, capture_output=True, text=True, shell=True)
    lines = [l for l in result.stdout.split('\n') if l]
    data = [line.split('|') for line in lines]
    df = pd.DataFrame(data, columns=["jobid", "start", "end", "elapsed", "state", "maxrss", "host"])

    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df['end'] = pd.to_datetime(df['end'], errors='coerce')

    df['maxrss'] = df['maxrss'].map(map_maxrss) / 1e9
    df["elapsed"] = pd.to_timedelta(df["elapsed"])

    # Collect info on pending tasks.
    # returns a list of all pending task IDs, no header (-h) and expanding batch tasks (-r) for me.
    cmd2 = rf'squeue --me --format="%20i" --state=PD -r -h'
    result2 = sp.run(cmd2, capture_output=True, text=True, shell=True)
    lines2 = [l for l in result2.stdout.split('\n') if l]
    data2 = [line.strip() for line in lines2]
    data2 = [[v + '.batch', 'PENDING'] for v in data2 if v.startswith(jobids_str)]
    df2 = pd.DataFrame(data2, columns=["jobid", "state"])
    df = pd.concat([df, df2], ignore_index=True)

    return df


def map_maxrss(value):
    value = str(value)
    if value == '':
        return 0
    suffix_factor = {
        'K': 2**10,
        'M': 2**20,
        'G': 2**30,
        'T': 2**40,
    }
    return float(value[:-1]) * suffix_factor[value[-1]]


def print_df(df, full):
    df_comp = df[df.state == 'COMPLETED']
    df_print = df[df.state != 'PENDING']

    if full:
        print(df_print.to_string())
        print()
    state_value_counts = df.state.value_counts()
    for state in ['PENDING', 'RUNNING', 'COMPLETED', 'CANCELLED', 'FAILED']:
        if state in state_value_counts:
            value = state_value_counts[state]
            print(f"{state:<10}: {value}")

    total = len(df)
    print(f"TOTAL     : {total}")

    print()
    print('earliest start:', df_comp.start.min())
    print('latest end    :', df_comp.end.max())
    print('total duration:', df_comp.end.max() - df_comp.start.min())

    print()
    varname = 'elapsed'
    print(f'Completed {varname}')
    for meth in ['min', 'mean', 'max']:
        print(f'  {meth:<5}: {getattr(df_comp[varname], meth)()}')
    varname = 'maxrss'
    print(f'Completed {varname}')
    for meth in ['min', 'mean', 'max']:
        print(f'  {meth:<5}: {getattr(df_comp[varname], meth)():.1f}G')

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
    jobids = [list(Path.cwd().glob('slurm/output/*' + j[:-6] + '*.err'))[0] for j in df.jobid.values.tolist()]
    sp.run(['vim'] + jobids)


# Useful in interactive mode.
def fastest_hosts(df, min_num_runs=3, nhosts=40):
    df_comp = df[df.state == 'COMPLETED']
    host_stats = df_comp.groupby('host')['elapsed'].agg(['mean', 'count'])
    host_stats = host_stats[host_stats['count'] >= min_num_runs]
    top_40 = host_stats.sort_values(by='mean').head(nhosts)
    print(top_40.to_string())

    node_string = ",".join(sorted(top_40.index.tolist()))
    print(f"#SBATCH --prefer={node_string}")


@click.command
@click.option('--full', '-F', is_flag=True)
@click.option('--interactive', '-I', is_flag=True)
@click.argument('jobids_str')
def main(full, interactive, jobids_str):
    df = run_parse_cmd(jobids_str)
    print_df(df, full)
    if interactive:
        breakpoint()