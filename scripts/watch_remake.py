#!/usr/bin/env python3
"""Watch this repo's remake SLURM arrays: print events, and push stage completions to ntfy.

Follows the current array job id of each rule via .remake/jobs/<rule>.jobids.json, so resubmissions are
picked up automatically. Only jobs with id >= --since are watched (ignores earlier test runs).

Printed events (one line each):
    DONE <rule> ...          whole array completed (also pushed to ntfy)
    FINISHED-WITH-FAILURES   array left the queue with failures
    FAIL <rule>[i] ...       an element failed (OOM/TIMEOUT/FAILED/NODE_FAIL/CANCELLED)
    BLOCKED <rule> ...       dependency can never be satisfied (an upstream task failed)
    SLOW <rule>[i] ...       element running much longer than the completed ones
    STALLED <rule>[i] ...    element's per-task log has not been written to for --stall minutes (the actionable
                             signal: packed nodes make tasks slow but they keep logging)
    PROGRESS ...             every --heartbeat minutes

Usage (repo root):  python3 scripts/watch_remake.py --since <first job id> [--topic hk26-jasmin-updates]
"""
import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RULES = ['create_stores', 'regrid'] + [f'coarsen_{d}_z{z}' for d in ('2d', '3d') for z in range(9, -1, -1)]
FAIL_STATES = {'FAILED', 'OUT_OF_MEMORY', 'TIMEOUT', 'NODE_FAIL', 'CANCELLED', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED'}
# Slow threshold: max(floor minutes, factor x median completed elapsed).
# Regrid tasks packed onto busy nodes routinely take 2-4x the median (seen: 150+ min vs 45 min median), and
# recover on their own, so SLOW is only for real outliers; STALLED (no log output) is what needs acting on.
SLOW = {'create_stores': (120, 4), 'regrid': (240, 4)}
SLOW_DEFAULT = (90, 4)
LOGDIR = Path('/work/scratch-nopw2/mmuetz/um2hp/logs/prod/v7/glm.n2560_RAL3p3_tuned_p4k')


def event(msg):
    print(f'{datetime.now():%H:%M} {msg}', flush=True)


def notify(topic, msg):
    event(f'NTFY {msg}')
    try:
        urllib.request.urlopen(urllib.request.Request(f'https://ntfy.sh/{topic}', data=msg.encode()), timeout=20)
    except Exception as e:
        event(f'NTFY-FAILED {e}')


def minutes(s):
    """SLURM [D-][HH:]MM:SS -> minutes."""
    days = 0
    if '-' in s:
        d, s = s.split('-')
        days = int(d)
    parts = [int(p) for p in s.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 1440 + h * 60 + m + sec / 60


def task_log_path(rule, idx):
    """Per-task log on scratch for array element idx of rule, via the rule's latest job spec file."""
    specs = sorted((REPO / '.remake' / 'jobs').glob(f'{rule}.[0-9]*.json'), key=lambda p: int(p.name.split('.')[-2]))
    if not specs:
        return None
    try:
        kwargs = json.loads(specs[-1].read_text())[idx]['kwargs']
    except (IndexError, KeyError, json.JSONDecodeError):
        return None
    if rule == 'regrid':
        return LOGDIR / 'regrid' / f"{datetime.fromisoformat(kwargs['date']):%Y%m%dT%H}.log"
    if rule.startswith('coarsen_'):
        _, dim, zoom = rule.split('_')
        return LOGDIR / 'coarsen' / dim / zoom / f"{kwargs['start']}.log"
    return LOGDIR / 'create_stores.log'


def log_idle_minutes(rule, idx):
    """Minutes since the task's log was last written (None if there is no log yet)."""
    path = task_log_path(rule, idx)
    if path is None or not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 60


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def array_state(jid):
    finished = {}  # idx -> (state, elapsed min, node)
    for line in run(['sacct', '-n', '-X', '-P', '-j', jid, '-o', 'JobID,State,Elapsed,NodeList']).splitlines():
        jobid, state, elapsed, node = line.split('|')
        if '_' not in jobid or '[' in jobid:
            continue
        state = state.split()[0]
        if state not in ('RUNNING', 'PENDING', 'REQUEUED'):
            finished[int(jobid.split('_')[1])] = (state, minutes(elapsed), node)
    queued = {}  # idx -> (state, reason, elapsed min, node)
    for line in run(['squeue', '-h', '-r', '-j', jid, '-o', '%i|%T|%r|%M|%N']).splitlines():
        jobid, state, reason, elapsed, node = line.split('|')
        if '_' in jobid:
            queued[int(jobid.split('_')[1])] = (state, reason, minutes(elapsed), node)
    return finished, queued


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--since', type=int, required=True)
    parser.add_argument('--topic', default='hk26-jasmin-updates')
    parser.add_argument('--interval', type=int, default=60, help='seconds between polls')
    parser.add_argument('--heartbeat', type=int, default=30, help='minutes between PROGRESS lines')
    parser.add_argument('--stall', type=int, default=75, help='minutes of no log output before STALLED')
    args = parser.parse_args()

    seen = set()
    done_jobs = set()
    first_pass = True  # arrays already finished when the watcher starts are marked done without notifying
    last_heartbeat = time.time()
    event(f'watching rules {RULES[0]}..{RULES[-1]} for job ids >= {args.since}')
    while True:
        progress = []
        for rule in RULES:
            jobids_file = REPO / '.remake' / 'jobs' / f'{rule}.jobids.json'
            if not jobids_file.exists():
                continue
            jid = str(json.loads(jobids_file.read_text())['slurm_array_job_id'])
            if int(jid) < args.since or jid in done_jobs:
                continue
            finished, queued = array_state(jid)
            if not finished and not queued:
                continue
            total = len(finished) + len(queued)
            ok_times = [f[1] for f in finished.values() if f[0] == 'COMPLETED']
            failed = {i: f for i, f in sorted(finished.items()) if f[0] in FAIL_STATES}

            for i, (state, elapsed, node) in failed.items():
                if (jid, i) not in seen:
                    seen.add((jid, i))
                    event(f'FAIL {rule}[{i}] {state} after {elapsed:.0f} min on {node} (job {jid}_{i})')

            if (jid, 'blocked') not in seen and any('DependencyNeverSatisfied' in q[1] for q in queued.values()):
                seen.add((jid, 'blocked'))
                event(f'BLOCKED {rule} (job {jid}): DependencyNeverSatisfied - an upstream task failed')

            floor, factor = SLOW.get(rule, SLOW_DEFAULT)
            threshold = max(floor, factor * statistics.median(ok_times)) if len(ok_times) >= 5 else 2 * floor
            for i, (state, reason, elapsed, node) in queued.items():
                if state == 'RUNNING' and elapsed > threshold and (jid, i, 'slow') not in seen:
                    seen.add((jid, i, 'slow'))
                    median = f'{statistics.median(ok_times):.0f}' if ok_times else '-'
                    event(f'SLOW {rule}[{i}] running {elapsed:.0f} min on {node} '
                          f'(median completed {median} min, threshold {threshold:.0f}; job {jid}_{i})')
                if state == 'RUNNING' and (jid, i, 'stalled') not in seen:
                    idle = log_idle_minutes(rule, i)
                    if idle is not None and idle > args.stall:
                        seen.add((jid, i, 'stalled'))
                        event(f'STALLED {rule}[{i}] no log output for {idle:.0f} min '
                              f'(running {elapsed:.0f} min on {node}; job {jid}_{i})')

            if not queued and first_pass:
                done_jobs.add(jid)
                event(f'already finished at start: {rule} (job {jid}), {len(ok_times)}/{total} completed')
                continue
            if not queued:
                done_jobs.add(jid)
                median = f', median {statistics.median(ok_times):.0f} min' if ok_times else ''
                if failed:
                    event(f'FINISHED-WITH-FAILURES {rule}: {len(ok_times)}/{total} completed, {len(failed)} failed '
                          f'(job {jid})')
                else:
                    notify(args.topic, f'✅ p4k {rule}: all {total} tasks completed (job {jid}{median})')
            else:
                running = sum(q[0] == 'RUNNING' for q in queued.values())
                progress.append(f'{rule} {len(ok_times)}/{total} done, {running} running, {len(failed)} failed')

        first_pass = False
        if progress and time.time() - last_heartbeat > args.heartbeat * 60:
            event('PROGRESS ' + '; '.join(progress))
            last_heartbeat = time.time()
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
