"""Purpose is to do a simple regrid for different sims/settings, and store the results to allow for benchmarking."""
import sys
from collections import defaultdict

import iris
from pathlib import Path
from timeit import default_timer as timer

from loguru import logger
import numpy as np
import pandas as pd
import xarray as xr

from um_to_healpix.latlon_to_healpix import gen_weights, LatLon2HealpixRegridder
from um_to_healpix.um_slurm_control import find_dyamond3_pp_dates_to_paths
from um_to_healpix.util import load_config
from um_to_healpix.cube_to_da_mapping import DataArrayExtractor
from um_to_healpix.um_process_tasks import regrid_da_to_healpix, weights_filename


class Benchmarker:
    def __init__(self, config_path, repeat=3):
        self.runidx = 0
        self.config_path = config_path
        self.config_obj = load_config(config_path)
        self.repeat = repeat

        self.output = defaultdict(list)

        self.sim = None
        self.config = None
        self.das = {}


    def load(self, sim):
        self.sim = sim
        self.config = self.config_obj.processing_config[sim]
        basedir = self.config['basedir']
        # weightsdir = config['weightsdir']
        weightsdir = Path('/work/scratch-nopw2/mmuetz/weightsdir')

        zoom = self.config['max_zoom']

        g2d = self.config['groups']['2d']
        mapping_key = ('psl', 'air_pressure_at_mean_sea_level')
        cube_map = g2d['name_map'][mapping_key]

        dates_to_paths = find_dyamond3_pp_dates_to_paths(basedir)
        first_date = sorted(dates_to_paths.keys())[0]
        inpaths = dates_to_paths[first_date]

        logger.info(f"Loading cubes for {first_date}...")
        cubes = iris.load(inpaths)

        # Standard extractor setup (p and z needed for 3D, but optional for 2D psl)
        extractor = DataArrayExtractor(None, None)
        da = extractor.extract_da(cube_map, cubes.extract(g2d['constraint']))

        lonname = [c for c in da.coords if c.startswith('longitude')][0]
        latname = [c for c in da.coords if c.startswith('latitude')][0]
        add_cyclic = self.config.get('add_cyclic', True)
        regional = self.config.get('regional', False)
        self.weights_path = weightsdir / weights_filename(da, zoom, lonname, latname, add_cyclic, regional)
        if not self.weights_path.exists():
            logger.info(f'No weights for {da.name}, generating')
            # chunks[-1] selects the spatial chunk.
            chunks = g2d['chunks'][zoom]
            gen_weights(da, weights_path=self.weights_path, zoom=zoom, lonname=lonname, latname=latname,
                        add_cyclic=add_cyclic, regional=regional, regional_chunks=chunks[-1])
        self.da = da.compute()
        self.drop_vars = self.config['drop_vars']

        self.kwargs = dict(
            zoom_level=zoom,
            add_cyclic=self.config.get('add_cyclic', True),
            regional=self.config.get('regional', False),
            weights = xr.load_dataset(self.weights_path),
        )

    def run(self, method, nproc=None):
        self.runidx += 1
        logger.info(f"{self.runidx=}, {method=}, {nproc=}, {self.repeat=}")
        da = self.da
        drop_vars_exists = list(set(self.drop_vars) & set(k for k in da.coords.keys()))
        da = da.drop_vars(drop_vars_exists)
        lonname = [c for c in da.coords if c.startswith('longitude')][0]
        latname = [c for c in da.coords if c.startswith('latitude')][0]

        for i in range(self.repeat):
            start = timer()

            regridder = LatLon2HealpixRegridder(method=method, nproc=nproc, **self.kwargs)
            da_hp = regridder.regrid(da, lonname, latname)

            end = timer()

            if self.sim not in self.das:
                self.das[self.sim] = da_hp
            else:
                if not np.isclose(self.das[self.sim].values, da_hp.values).all():
                    breakpoint()

            self.output['runidx'].append(self.runidx)
            self.output['repidx'].append(i)
            self.output['sim'].append(self.sim)
            self.output['nproc'].append(nproc)
            self.output['method'].append(method)
            self.output['time'].append(end - start)

    def disp_output(self):
        df = pd.DataFrame(self.output)
        print(df)
        df_summary = df.groupby('runidx').agg({
            'sim': 'first',
            'method': 'first',
            'nproc': 'first',
            'time': ['min', 'max', 'mean'],
        })
        print(df_summary)

        breakpoint()


def main(config_path, sims):
    benchmark = Benchmarker(config_path, 5)
    for sim in sims:
        benchmark.load(sim)
        benchmark.run('easygems_delaunay')
        for i, nproc in enumerate([1, 2, 4, 6, 8, 12]):
            benchmark.run('easygems_delaunay_parallel', nproc)
    benchmark.disp_output()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python regrid_benchmark.py <config_path> <sim>")
        sys.exit(1)
    config_path = Path(sys.argv[1])
    sims = sys.argv[2:]
    main(config_path, sims)
