import iris
import matplotlib.pyplot as plt
import numpy as np

import um_to_healpix.util


def plot_zonal_mean(pr, zonal_mean, lat):
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
    pr_time_mean = pr.data.mean(axis=0)
    weights = np.cos(np.deg2rad(lat))
    weights_2d = weights[:, np.newaxis] * np.ones(pr.data.shape[2])[np.newaxis, :]
    zonal_mean = np.sum(pr_time_mean * weights_2d, axis=1) / np.sum(weights_2d, axis=1)
    return zonal_mean


def load_data(sim_config):
    cubes = iris.load(sorted((sim_config['basedir'] / 'field.pp/apverb.pp').glob('*.pp'))[0])

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

    pr = rain.copy()
    pr.data += snow.data
    return dict(rain=rain, snow=snow, pr=pr)


def main(sim, config):
    sim_config = config.processing_config[sim]

    precips = load_data(sim_config)
    for name, cube in precips.items():
        print(name)
        lat = cube.coord('latitude').points
        zonal_mean = calc_zonal_mean(cube, lat)
        plot_zonal_mean(cube, zonal_mean, lat)
        plt.savefig(f'/home/users/mmuetz/tmp/{name}_zonal_mean.png')


if __name__ == '__main__':
    sim = 'glm.n2560_RAL3p3.tuned'
    config = um_to_healpix.util.load_config('../config/hk26_config.py')
    main(sim, config)

