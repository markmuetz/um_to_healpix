"""Compare two deployments of the same sim's healpix zarr stores (e.g. dev_remake vs prod).

For each variable in each store (PT1H/PT3H) at one zoom, compares a sample of time steps (and pressure levels)
and reports NaN-pattern mismatches and max absolute/relative differences, flagging anything above rtol.
Also reports which time steps of store A have been written (chunk objects present), to check coverage.

Usage (repo root, pixi env):
    pixi run python scripts/compare_stores.py glm.n2560_RAL3p3.tuned --a dev_remake --b prod --zoom 10 \
        --times 2020-01-20T06 2020-01-24T13 --levels 1000 500 100

With --b-key, B is a different sim (e.g. a perturbed run vs its control): structure, coverage and NaN patterns
should still match, and meanDiff (A - B) shows whether the value differences are plausible.

N.B. float32 has a relative precision of ~6e-8 (1 ulp), so rtol=1e-7 means "at most ~1-2 ulp".
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from um_to_healpix.um_process_tasks import get_jasmin_s3
from um_to_healpix.util import load_config

IGNORE_ATTRS = {'deploy', 'processing_version'}


def open_store(url):
    import s3fs
    return xr.open_zarr(s3fs.S3Map(root=url, s3=get_jasmin_s3(), check=False), consolidated=True, chunks=None)


def written_times(fs, url, var, ntime, time_chunk=1):
    """Time indices covered by chunks for which every spatial (and level) chunk of var exists.

    Zarr keys index *chunks*, not time steps: below z9 a chunk spans many time steps (time_chunk), so a chunk
    index must be expanded to the steps it covers, or coverage is under-reported by that factor.
    """
    keys = fs.ls(f'{url[5:]}/{var}', detail=False)
    counts = defaultdict(int)
    for key in keys:
        name = key.rsplit('/', 1)[-1]
        if re.fullmatch(r'\d+(\.\d+)+', name):
            counts[int(name.split('.')[0])] += 1
    if not counts:
        return []
    full = max(counts.values())
    times = []
    for chunk, n in counts.items():
        if n == full:
            times.extend(range(chunk * time_chunk, min((chunk + 1) * time_chunk, ntime)))
    return sorted(t for t in times if t < ntime)


def compare_field(a, b, rtol):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    both = ~nan_a & ~nan_b
    diff = np.abs(a[both] - b[both])
    denom = np.maximum(np.abs(b[both]), np.finfo(np.float32).tiny)
    rel = diff / denom
    return {
        'mean_diff': float((a[both] - b[both]).mean()) if both.any() else 0.0,
        'n': int(a.size),
        'nan_a': int(nan_a.sum()),
        'nan_b': int(nan_b.sum()),
        'nan_mismatch': int((nan_a != nan_b).sum()),
        'n_diff': int((diff > 0).sum()),
        'max_abs': float(diff.max()) if diff.size else 0.0,
        'max_rel': float(rel.max()) if rel.size else 0.0,
        'argmax_rel': int(np.flatnonzero(both)[rel.argmax()]) if rel.size and rel.max() > 0 else None,
        'n_over_rtol': int((rel > rtol).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config_key')
    parser.add_argument('--a', default='dev_remake', help='deploy of store A (checked for coverage)')
    parser.add_argument('--b', default='prod', help='deploy of store B (reference)')
    parser.add_argument('--b-key', default=None, help='config_key of store B (default: same as A)')
    parser.add_argument('--zoom', type=int, default=10)
    parser.add_argument('--times', nargs='+', required=True, help='ISO times to compare (nearest available)')
    parser.add_argument('--levels', nargs='+', type=float, default=[1000, 500, 100], help='pressure levels (hPa)')
    parser.add_argument('--rtol', type=float, default=1e-7)
    parser.add_argument('--vars', nargs='*', default=None)
    parser.add_argument('--config', default='config/hk26_config.py')
    args = parser.parse_args()

    config = load_config(Path(args.config))
    tpl = config.processing_config[args.config_key]['zarr_store_url_tpl']
    fs = get_jasmin_s3()
    flagged = []

    for freq in ['PT1H', 'PT3H']:
        url_a = tpl.format(freq=freq, zoom=args.zoom).replace(f'/{config.deploy}/', f'/{args.a}/')
        tpl_b = config.processing_config[args.b_key or args.config_key]['zarr_store_url_tpl']
        url_b = tpl_b.format(freq=freq, zoom=args.zoom).replace(f'/{config.deploy}/', f'/{args.b}/')
        print(f'\n=== {freq} z{args.zoom}\n  A: {url_a}\n  B: {url_b}')
        ds_a, ds_b = open_store(url_a), open_store(url_b)

        # Structure.
        vars_a, vars_b = set(ds_a.data_vars), set(ds_b.data_vars)
        if vars_a != vars_b:
            flagged.append(f'{freq}: data_vars differ: only A {sorted(vars_a - vars_b)}, only B {sorted(vars_b - vars_a)}')
        for coord in ['time', 'healpix_index', 'pressure']:
            if coord in ds_a.coords or coord in ds_b.coords:
                if not (coord in ds_a.coords and coord in ds_b.coords and ds_a[coord].equals(ds_b[coord])):
                    flagged.append(f'{freq}: coord {coord} differs')
        ignore_attrs = IGNORE_ATTRS | ({'simulation'} if args.b_key else set())
        attr_diff = {k for k in set(ds_a.attrs) | set(ds_b.attrs)
                     if k not in ignore_attrs and ds_a.attrs.get(k) != ds_b.attrs.get(k)}
        if attr_diff:
            flagged.append(f'{freq}: global attrs differ: {sorted(attr_diff)}')

        times = pd.DatetimeIndex(ds_a.time.values)
        tidx = sorted({int(times.get_indexer([pd.Timestamp(t)], method='nearest')[0]) for t in args.times})
        variables = [v for v in sorted(vars_a & vars_b) if 'time' in ds_a[v].dims
                     and (args.vars is None or v in args.vars)]

        # Coverage of A (which time steps are fully written).
        print('  coverage of A (fully written time steps):')
        for var in variables:
            chunks = ds_a[var].encoding.get('chunks') or (1,)
            wt = written_times(fs, url_a, var, len(times), time_chunk=chunks[0])
            span = f'{times[wt[0]]:%Y-%m-%dT%H} .. {times[wt[-1]]:%Y-%m-%dT%H}' if wt else '-'
            gaps = (wt[-1] - wt[0] + 1 - len(wt)) if wt else 0
            print(f'    {var:8s} {len(wt):5d} steps  {span}  gaps={gaps}')
            if gaps:
                flagged.append(f'{freq} {var}: {gaps} gaps in written time steps of A')

        # Values.
        print(f'  values at {[f"{times[i]:%Y-%m-%dT%H}" for i in tidx]}:')
        hdr = f'    {"var":8s} {"time":13s} {"lev":>6s} {"nanA":>9s} {"nanB":>9s} {"nanMis":>7s} ' \
              f'{"nDiff":>9s} {"maxAbs":>10s} {"maxRel":>10s} {">rtol":>7s} {"meanDiff":>10s}'
        print(hdr)
        for var in variables:
            if (ds_a[var].attrs.get('units'), ds_a[var].attrs.get('standard_name')) != \
                    (ds_b[var].attrs.get('units'), ds_b[var].attrs.get('standard_name')):
                flagged.append(f'{freq} {var}: units/standard_name differ')
            levels = [None]
            if 'pressure' in ds_a[var].dims:
                levels = [lev for lev in args.levels if lev in ds_a.pressure.values]
            for i in tidx:
                for lev in levels:
                    sel = {'time': i}
                    da_a = ds_a[var].isel(sel)
                    da_b = ds_b[var].isel(sel)
                    if lev is not None:
                        da_a, da_b = da_a.sel(pressure=lev), da_b.sel(pressure=lev)
                    r = compare_field(da_a.values, da_b.values, args.rtol)
                    lev_str = '' if lev is None else f'{lev:g}'
                    print(f'    {var:8s} {times[i]:%Y-%m-%dT%H} {lev_str:>6s} {r["nan_a"]:9d} {r["nan_b"]:9d} '
                          f'{r["nan_mismatch"]:7d} {r["n_diff"]:9d} {r["max_abs"]:10.3g} {r["max_rel"]:10.3g} '
                          f'{r["n_over_rtol"]:7d} {r["mean_diff"]:10.3g}', flush=True)
                    if r['nan_mismatch'] or r['n_over_rtol']:
                        where = ''
                        if r['argmax_rel'] is not None:
                            import healpix
                            lon, lat = healpix.pix2ang(2 ** args.zoom, int(ds_a.healpix_index.values[r['argmax_rel']]),
                                                       nest=True, lonlat=True)
                            where = f' (max rel diff at lon={float(lon):.2f}, lat={float(lat):.2f})'
                        flagged.append(f'{freq} {var} {times[i]:%Y-%m-%dT%H} {lev_str}: nan_mismatch={r["nan_mismatch"]}, '
                                       f'{r["n_over_rtol"]}/{r["n"]} cells over rtol={args.rtol:g}, '
                                       f'max_rel={r["max_rel"]:.3g}, max_abs={r["max_abs"]:.3g}{where}')

    print(f'\n=== FLAGGED ({len(flagged)}):')
    for f in flagged:
        print('  ' + f)
    if not flagged:
        print(f'  none: all compared fields identical in NaN pattern and within rtol={args.rtol:g}')


if __name__ == '__main__':
    main()
