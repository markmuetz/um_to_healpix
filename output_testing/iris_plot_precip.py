# coding: utf-8
import iris
import matplotlib.pyplot as plt
import numpy as np
import um_to_healpix.util
import matplotlib.pyplot as plt


sim = 'glm.n2560_RAL3p3.tuned'
config = um_to_healpix.util.load_config('../config/hk26_config.py')
sim_config = config.processing_config[sin]

cubes = iris.load(sorted((sim_config['basedir'] / 'field.pp/apverb.pp').glob('*.pp'))[0])
for i, cube in enumerate(cubes):
    print(i, repr(cube))

rain = cubes[24]
snow = cubes[28]
pr = rain.copy()
pr.data += snow.data

lat = pr.coord('latitude').points

pr_time_mean = pr.data.mean(axis=0)
weights = np.cos(np.deg2rad(lat))
weights_2d = weights[:, np.newaxis] * np.ones(pr.data.shape[2])[np.newaxis, :]
zonal_mean = np.sum(pr_time_mean * weights_2d, axis=1) / np.sum(weights_2d, axis=1)


time_coord = cube.coord('time')
times = time_coord.units.num2date(time_coord.points)
t1 = times[0]
t2 = times[-1]

plt.figure(layout='constrained')
plt.title(f'{t1} - {t2}')

plt.plot(lat, zonal_mean)

plt.xlim((-90, 90))
plt.xlabel('lat')
plt.ylabel(f'precipitation_flux ({pr.units})')

plt.savefig('/home/users/mmuetz/tmp/pr_zonal_mean.png')
