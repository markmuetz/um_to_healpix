"""um2hp: inspect the processing config, validate it against the input .pp files, and report pipeline status.

This is what is left of um_slurm_control.py after the remake3 migration. Job submission is remake's job now
(`remake run remakefile_regrid.py -E slurm`), so the submission commands are gone; what remains is the part
remake has no opinion about - does the config for a given simulation actually match its input files - plus a
cross-remakefile status view, which remake's own per-remakefile CLI cannot give.

    um2hp ls                                  # config keys
    um2hp print-config <key> [dict keys...]   # drill into one config
    um2hp check-output-mapping <key|all>      # validate config against the .pp files
    um2hp analyse-output-mapping -i out.csv   # diff that across simulations
    um2hp status [<key>]                      # regrid + coarsen + checks, one table
"""
import pprint
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from .util import load_config

REGRID_REMAKEFILE = 'remakefile_regrid.py'
COARSEN_REMAKEFILE = 'remakefile_coarsen.py'
# Level coordinate each group is expected to carry, used to flag a variable mapped into the wrong group.
GROUP_LEVEL_COORD = {
    '2d': None,
    '2d_depth': 'depth',
    '3d': 'pressure',
    '3d_ml': 'model_level_number',
}
CHECK_COLS = ['expt', 'store', 'group', 'short_name', 'standard_name', 'units', 'present', 'cube_name',
              'stash_code', 'levels', 'grid', 'weights', 'extra_attrs', 'extra_processing']


@click.group()
@click.option('--config', '-C', default=Path('config/hk26_config.py'), type=Path)
@click.option('--debug', '-D', is_flag=True)
@click.pass_context
def cli(ctx, config, debug):
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = str(config)
    ctx.obj['config'] = load_config(config)
    if not debug:
        logger.remove()
        logger.add(lambda m: click.echo(m, err=True), level='INFO', format='{message}')


@cli.command()
@click.pass_context
def ls(ctx):
    """List the config keys."""
    for key in ctx.obj['config'].processing_config:
        print(key)


@cli.command('print-config')
@click.argument('args', nargs=-1)
@click.option('--list-keys', '-L', is_flag=True)
@click.pass_context
def print_config(ctx, list_keys, args):
    """Print one simulation's config, drilling in with successive dict keys."""
    config = ctx.obj['config'].processing_config[args[0]]
    for dict_key in args[1:]:
        try:
            config = config[dict_key]
        except KeyError:
            print(f'Possible keys are: {", ".join(config.keys())}')
            raise
    pprint.pprint(list(config) if list_keys else config)


class _FakeCoord:
    """Just enough of an xarray coord for weights_filename: .values and len()."""

    def __init__(self, points):
        self.values = points

    def __len__(self):
        return len(self.values)


def _grid_of(cube, max_zoom):
    """(human-readable grid, weights filename) for a cube, or (None, None) if it has no lat/lon.

    Goes through the real weights_filename so this cannot drift from what the pipeline looks for.
    """
    from .um_process_tasks import weights_filename
    try:
        lon, lat = cube.coord('longitude'), cube.coord('latitude')
    except Exception:
        return None, None
    lo, la = lon.points, lat.points
    grid = f'{len(lo)}x{len(la)} lon {lo[0]:.3f}..{lo[-1]:.3f} lat {la[0]:.3f}..{la[-1]:.3f}'
    fake = {'longitude': _FakeCoord(lo), 'latitude': _FakeCoord(la)}
    return grid, weights_filename(fake, max_zoom, 'longitude', 'latitude', True, False)


def _at_level(cube, level_coord):
    """Does this cube sit at the level the group expects? ('' / None means a surface field.)"""
    present = {c.name() for c in cube.coords() if c.name() in LEVEL_COORDS}
    return level_coord in present if level_coord else not present


def _extract_first(map_item, cubes, level_coord):
    """First raw cube matching each constituent of a MapItem/MultiMapItem, at the group's level.

    extract_cubes() uses extract_cube(), which demands exactly one match; load_raw gives one cube per
    (time, level), so it would always raise. Grid, STASH and units are the same for every step, so take the
    first -- but filter by level first, because the group constraint (has_dimensions(..., 'pressure', ...))
    cannot be applied to raw cubes and several variables are only distinguishable by it: ua and uas are both
    x_wind / m01s03i225, as are va/vas, ta/tas and hus/huss. Without this the 3d rows report the 2d cube's
    grid, which is a different grid and so a different weights file.

    Returns (cubes, note) where note explains an empty result.
    """
    from .cube_to_da_mapping import MultiMapItem
    items = map_item.items if isinstance(map_item, MultiMapItem) else [map_item]
    out = []
    for item in items:
        matched = cubes.extract(item.iris_constraint)
        if not matched:
            return [], 'no cube matches the constraint'
        at_level = [c for c in matched if _at_level(c, level_coord)]
        if not at_level:
            found = sorted({l for c in matched for l in
                            ({x.name() for x in c.coords()} & set(LEVEL_COORDS))}) or ['(surface)']
            return [], f'matched {len(matched)} cube(s) but none at {level_coord or "(surface)"}; found {found}'
        out.append(at_level[0])
    return out, None


LEVEL_COORDS = ('depth', 'pressure', 'model_level_number')


def _levels_of(cube):
    """Which level coordinate(s) the cube carries.

    Checks every coord, not just dim coords: load_raw gives one cube per (time, level), so a model-level field
    has model_level_number as a *scalar* coord and would otherwise look level-less.
    """
    names = [c.name() for c in cube.coords() if c.name() in LEVEL_COORDS]
    return ','.join(names) or '-'


@cli.command('check-output-mapping')
@click.argument('config_key')
@click.option('--date', '-d', default=None)
@click.option('--output-file', '-o', default=None)
@click.pass_context
def check_output_mapping(ctx, config_key, date, output_file):
    """Check every configured output variable resolves against the real .pp files.

    Reports, per variable: whether it is present, which cube/STASH it came from, which grid it sits on and
    whether a weights file for that grid already exists (missing weights mean create_stores must generate one,
    which needs 30-40 GB and hours), and whether its level coordinate matches the group it is mapped into.

    Uses iris.load_raw: field headers only, no merge and no data read.
    """
    import iris
    import operator

    from .pp_scan import DEFAULT_PP_GLOB, find_dyamond3_pp_dates_to_paths

    op_symbol = {operator.add: '+', operator.sub: '-', operator.mul: '*', operator.truediv: '/'}
    full_config = ctx.obj['config']
    config_keys = list(full_config.processing_config) if config_key == 'all' else [config_key]

    rows, mismatched = [], []
    for key in config_keys:
        logger.info(f'check output mapping: {key}')
        config = full_config.processing_config[key]
        weightsdir = Path(config['weightsdir'])
        # The +4K reruns use a different layout; honour the config rather than the default glob.
        dates_to_paths = find_dyamond3_pp_dates_to_paths(config['basedir'],
                                                         config.get('pp_glob', DEFAULT_PP_GLOB))
        if not dates_to_paths:
            logger.error(f'{key}: no .pp files under {config["basedir"]} '
                         f'(glob {config.get("pp_glob", DEFAULT_PP_GLOB)!r})')
            continue
        want = pd.Timestamp(date) if date else config['first_date']
        if want not in dates_to_paths:
            logger.error(f'{key}: no input files for {want}; first available is {min(dates_to_paths)}')
            continue
        cubes = iris.load_raw([str(p) for p in dates_to_paths[want]])
        logger.info(f'  {len(cubes)} raw fields from {len(dates_to_paths[want])} streams')

        for group_name, group in config['groups'].items():
            expected_level = GROUP_LEVEL_COORD.get(group_name, 'unknown')
            for (short_name, long_name), map_item in group['name_map'].items():
                extra_processing = map_item.extra_processing
                if extra_processing and not isinstance(extra_processing, str):
                    extra_processing = extra_processing.__name__
                row = dict(expt=key, store=group['zarr_store'], group=group_name, short_name=short_name,
                           standard_name=long_name, units=None, present=False, cube_name=None, stash_code=None,
                           levels=None, grid=None, weights=None, extra_attrs=map_item.extra_attrs,
                           extra_processing=extra_processing)
                level_coord = expected_level if expected_level != 'unknown' else None
                try:
                    item_cubes, note = _extract_first(map_item, cubes, level_coord)
                except iris.exceptions.ConstraintMismatchError as e:
                    item_cubes, note = [], str(e)
                if not item_cubes:
                    logger.error(f'  {key}/{short_name} ({group_name}): {note}')
                    mismatched.append(f'{key}/{short_name}')
                    rows.append(row)
                    continue
                cube = item_cubes[0]
                if len(item_cubes) == 1:
                    cube_name, stash = cube.name(), str(cube.attributes.get('STASH'))
                else:
                    names, stashes = [cube.name()], [str(cube.attributes.get('STASH'))]
                    for op, nxt in zip(map_item.ops, item_cubes[1:]):
                        names.extend([op_symbol.get(op, '?'), nxt.name()])
                        stashes.extend([op_symbol.get(op, '?'), str(nxt.attributes.get('STASH'))])
                    cube_name, stash = ' '.join(names), ' '.join(stashes)
                grid, weights_fn = _grid_of(cube, config['max_zoom'])
                levels = _levels_of(cube)
                row.update(units=map_item.units if map_item.units is not None else str(cube.units),
                           present=True, cube_name=cube_name, stash_code=stash, levels=levels, grid=grid,
                           weights=('present' if weights_fn and (weightsdir / weights_fn).exists()
                                    else 'MISSING' if weights_fn else None))
                rows.append(row)

    df = pd.DataFrame(rows, columns=CHECK_COLS)
    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f'written {output_file}')
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                               'display.width', 250, 'display.max_colwidth', 38):
            print(df.drop(columns=['extra_attrs', 'extra_processing', 'standard_name']).to_string(index=False))
    missing = df[~df.present]
    if len(missing):
        logger.error(f'{len(missing)} variable(s) NOT FOUND: '
                     f'{", ".join(missing.expt + "/" + missing.short_name)}')
    need = sorted(set(df[df.weights == 'MISSING'].grid.dropna()))
    if need:
        logger.warning(f'{len(need)} grid(s) have no weights file; create_stores will generate them '
                       f'(30-40 GB and hours each):')
        for g in need:
            logger.warning(f'    {g}')
    else:
        logger.info('all grids already have weights')
    if len(missing):
        raise SystemExit(1)


def _title(msg):
    # flush: logger writes to stderr, so without this the two streams interleave out of order.
    print(msg, flush=True)
    print('=' * len(msg), flush=True)


@cli.command('analyse-output-mapping')
@click.option('--input-file', '-i', required=True)
@click.pass_context
def analyse_output_mapping(ctx, input_file):
    """Compare a check-output-mapping CSV across simulations, highlighting what differs."""
    df = pd.read_csv(input_file)
    n = len(df.expt.unique())
    comparison_cols = df.drop(columns='expt')
    duplicate_counts = comparison_cols.groupby(comparison_cols.columns.tolist(), dropna=False).size()
    df_with_counts = df.merge(duplicate_counts.rename('counts'), how='left',
                              left_on=comparison_cols.columns.tolist(), right_index=True)

    _title('The same across all expts:')
    same = df_with_counts.counts == n
    print(df[same][~df_with_counts[same].drop(columns='expt').duplicated()].drop(columns='expt'))

    interesting = set(df_with_counts[df_with_counts.counts < n]['short_name'].unique())
    interesting.update(df_with_counts[~df_with_counts.present]['short_name'].unique())
    for var in sorted(interesting):
        _title(f'Interesting var: {var}')
        print(df[df.short_name == var])


@cli.command()
@click.argument('config_key', required=False)
@click.pass_context
def status(ctx, config_key):
    """Regrid, coarsen and check/plot progress for a simulation, in one table.

    remake's own CLI works one remakefile at a time; this joins them, which is the whole-pipeline view.
    """
    from remake import load_remake

    full = ctx.obj['config']
    scoped = list(getattr(full, 'remake_config_keys', full.processing_config))
    keys = [config_key] if config_key else scoped
    for key in keys:
        _title(f'{key}')
        if key not in full.processing_config:
            logger.error(f'{key} is not in the processing config; known keys: {", ".join(full.processing_config)}')
            continue
        if key not in scoped:
            # Otherwise this prints a bare 0/0, which reads as "nothing done" rather than "not built".
            logger.warning(f'{key} is not in config.remake_config_keys ({", ".join(scoped)}), so the '
                           f'remakefiles build no tasks for it. Add it there to see its status.')
            continue
        grand = dict(tasks=0, up_to_date=0, stale=0, failed=0, pending=0)
        for remakefile in (REGRID_REMAKEFILE, COARSEN_REMAKEFILE):
            if not Path(remakefile).exists():
                logger.warning(f'{remakefile} not found - skipping')
                continue
            rmk = load_remake(remakefile)
            summary = rmk.status_summary(query=f'config_key == "{key}"')
            print(f'  {remakefile}')
            print(f'    {"rule":<18}{"tasks":>7}{"done":>7}{"stale":>7}{"failed":>8}{"pending":>9}')
            for r in summary['rules']:
                if r.get('deferred'):
                    print(f'    {r["rule"]:<18}{"(deferred)":>38}')
                    continue
                if not r['tasks']:
                    continue
                print(f'    {r["rule"]:<18}{r["tasks"]:>7}{r["up_to_date"]:>7}{r["stale"]:>7}'
                      f'{r["failed"]:>8}{r["pending"]:>9}')
            t = summary['totals']
            for k in grand:
                grand[k] += t.get(k, 0)
        done, total = grand['up_to_date'], grand['tasks']
        pct = f'{100 * done / total:.1f}%' if total else 'n/a'
        print(f'  TOTAL {done}/{total} up to date ({pct}), {grand["stale"]} stale, '
              f'{grand["failed"]} failed, {grand["pending"]} pending')


if __name__ == '__main__':
    cli(obj={})
