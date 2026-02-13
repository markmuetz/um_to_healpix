"""Iris-based precipitation extraction and zonal-mean plotting helper.

Purpose:
- Load a representative UM PP file for one simulation.
- Extract stratiform rainfall and snowfall cubes.
- Build combined precipitation (rain + snow).
- Plot/save latitude-wise weighted zonal means.

This script is intentionally lightweight and diagnostic-oriented, and it uses
Iris directly rather than the full xarray HEALPix output path.
"""

import iris
import matplotlib.pyplot as plt
import numpy as np

import um_to_healpix.util


def plot_zonal_mean(pr, zonal_mean, lat):
    """Plot zonal mean precipitation vs latitude for one cube/time range.

    Args:
        pr: Iris cube used for title/units metadata.
        zonal_mean: 1D zonal-mean array across latitude.
        lat: 1D latitude coordinate values in degrees.
    """
    time_coord = pr.coord('time')
    times = time_coord.units.num2date(time_coord.points)
    t1 = times[0]
    t2 = times[-1]

    plt.figure(layout='constrained')
    plt.title(f'{t1} - {t2}')

    plt.plot(lat, zonal_mean)

    plt.xlim((-90, 90))
    plt.xlabel('lat')
    plt.ylabel(f'precipitation_flux ({pr.units})')


def calc_zonal_mean(pr, lat):
    """Compute cosine-latitude weighted zonal mean from a precipitation cube.

    Method:
    - Average over time first to get a 2D lat/lon slice.
    - Apply cosine(latitude) area weighting.
    - Collapse weighted field over longitude for each latitude.
    """
    pr_time_mean = pr.data.mean(axis=0)
    weights = np.cos(np.deg2rad(lat))
    weights_2d = weights[:, np.newaxis] * np.ones(pr.data.shape[2])[np.newaxis, :]
    zonal_mean = np.sum(pr_time_mean * weights_2d, axis=1) / np.sum(weights_2d, axis=1)
    return zonal_mean


def load_data(sim_config):
    """Load first PP file and extract rain/snow/combined precipitation cubes.

    Selection logic:
    - Uses non-empty cell-method variants of stratiform rain and snowfall flux.
    - Returns individual and combined cubes for comparative diagnostics.
    """
    # Load the earliest available PP file for quick representative diagnostics.
    cubes = iris.load(sorted((sim_config['basedir'] / 'field.pp/apverb.pp').glob('*.pp'))[0])

    # Extract rain and snow components using variable name + cell-method filter.
    rain = cubes.extract_cube(
        iris.Constraint(name='stratiform_rainfall_flux') &
        iris.Constraint(cube_func=um_to_healpix.util.cube_cell_method_is_not_empty)
    )
    snow = cubes.extract_cube(
        iris.Constraint(name='stratiform_snowfall_flux') &
        iris.Constraint(cube_func=um_to_healpix.util.cube_cell_method_is_not_empty)
    )
    print(repr(rain))
    print(repr(snow))

    # Combine to total precip while preserving cube metadata layout.
    pr = rain.copy()
    pr.data += snow.data
    return dict(rain=rain, snow=snow, pr=pr)


def main(sim, config):
    """Run extraction + zonal-mean plotting workflow for one simulation key."""
    sim_config = config.processing_config[sim]

    precips = load_data(sim_config)
    for name, cube in precips.items():
        print(name)
        # Compute and save one zonal-mean figure per precipitation variant.
        lat = cube.coord('latitude').points
        zonal_mean = calc_zonal_mean(cube, lat)
        plot_zonal_mean(cube, zonal_mean, lat)
        plt.savefig(f'/home/users/mmuetz/tmp/{name}_zonal_mean.png')


if __name__ == '__main__':
    # Default quick-run target.
    sim = 'glm.n2560_RAL3p3.tuned'
    config = um_to_healpix.util.load_config('../config/hk26_config.py')
    main(sim, config)

