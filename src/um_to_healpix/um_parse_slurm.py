import subprocess as sp
import sys

import pandas as pd


def run_parse_cmd(jobids_str):
    cmd = rf"sacct -P -o 'jobid%20,start,end,elapsed,state,MaxRSS,NodeList' -j {jobids_str}|grep -E '^[0-9_]*\.batch\|' --color=Never"
    result = sp.run(cmd, capture_output=True, text=True, shell=True)
    lines = [l for l in result.stdout.split('\n') if l]
    data = [line.split('|') for line in lines]
    df = pd.DataFrame(data, columns=["jobid", "start", "end", "elapsed", "state", "maxrss", "host"])

    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df['end'] = pd.to_datetime(df['end'], errors='coerce')

    df['maxrss'] = df['maxrss'].map(map_maxrss) / 1e9
    df["elapsed"] = pd.to_timedelta(df["elapsed"])

    # TODO: would be good to display info on pending tasks too.
    # cmd2 = rf'squeue --me --format="%20i" --state=PD -r -h'
    # result2 = sp.run(cmd2, capture_output=True, text=True, shell=True)
    # lines2 = [l for l in result2.stdout.split('\n') if l]
    # data2 = [[v.strip() for v in line.split('|')] for line in lines2]
    # breakpoint()
    # df2 = pd.DataFrame(data2, columns=["jobid"])
    # print(df2)

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


def print_df(df):
    df_comp = df[df.state == 'COMPLETED']

    print(df.to_string())
    print()
    for state, value in df.state.value_counts().items():
        print(f"{state:<10}: {value}")

    print()
    print('earliest start:', df_comp.start.min())
    print('latest end    :', df_comp.end.max())
    print('total duration:', df_comp.end.max() - df_comp.start.min())

    print()
    varname = 'elapsed'
    print(varname)
    for meth in ['min', 'mean', 'max']:
        print(f'  {meth:<5}: {getattr(df_comp[varname], meth)()}')
    varname = 'maxrss'
    print(varname)
    for meth in ['min', 'mean', 'max']:
        print(f'  {meth:<5}: {getattr(df_comp[varname], meth)():.1f}G')


def main():
    jobids_str = sys.argv[1]
    df = run_parse_cmd(jobids_str)
    print_df(df)