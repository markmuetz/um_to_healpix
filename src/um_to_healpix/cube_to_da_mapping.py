"""Helper classes that allow for a declarative specification of how to map from iris cubes to xarray DataArrays
"""
import os

import xarray as xr

from .util import model_level_to_pressure


class DataArrayExtractor:
    """Extract a cube or DataArray from a set of input cubes."""
    def __init__(self, p, z, nproc=None):
        self.p = p
        self.z = z
        # Vertical interpolation is the dominant cost of a regrid task and is GIL-bound, so it needs processes,
        # not the threads the regrid itself uses. Default to the CPUs the job asked for.
        #
        # UM2HP_INTERP_NPROC overrides it independently of SLURM_CPUS_PER_TASK. Both knobs read the same variable
        # otherwise, so a sweep of SLURM_CPUS_PER_TASK moves the threaded regrid (all 39 variables) at the same
        # time as this, and cannot attribute the difference to either. Keep them separable for measurement, and
        # in case processes (memory-hungry) and threads (not) ever want different values.
        if nproc is None:
            nproc = os.environ.get('UM2HP_INTERP_NPROC') or os.environ.get('SLURM_CPUS_PER_TASK', 1)
        self.nproc = int(nproc)

    @staticmethod
    def extract_cubes(map_item, group_cubes):
        """Extract an individual cube from a set of input cubes, possibly combining multiple cubes.

        Uses the constraints in map_item to extract the desired cube."""
        # Extract the cube and combine if necessary.
        if isinstance(map_item, MapItem):
            cube = group_cubes.extract_cube(map_item.iris_constraint)
            return [cube]
        elif isinstance(map_item, MultiMapItem):
            constraint_cubes = [group_cubes.extract_cube(map_item.iris_constraint) for map_item in map_item.items]
            return constraint_cubes
        else:
            raise Exception(f'unknown type {type(map_item)}')

    def _combined_cube(self, map_item, group_cubes):
        """The cube for this map item: extracted, shortened to 12 times, and combined if there are several."""
        # *Always* shorten cubes of time 13 to length 12 by ignoring first value.
        # This applies to the first cube of each day, for certain fields.
        cubes = self.extract_cubes(map_item, group_cubes)
        for i in range(len(cubes)):
            cube = cubes[i]
            if cube.shape[0] == 13:
                cubes[i] = cube[1:]

        cube = cubes[0]
        if isinstance(map_item, MultiMapItem) and len(cubes) > 1:
            for next_cube, op in zip(cubes[1:], map_item.ops):
                cube.data = op(cube.data, next_cube.data)
        return cube

    @staticmethod
    def _to_da(cube):
        # For some cubes (ones with names like m01s30i461 the da gets a name like filled-XXXXXX...
        # Make sure it's got the actual cube name so I can rename it later.
        return xr.DataArray.from_iris(cube).rename(cube.name())

    def is_model_level(self, map_item):
        return map_item.extra_processing == 'interpolate_model_levels_to_pressure'

    def extract_da(self, map_item, group_cubes):
        """Extracts a DataArray from a map item and group of cubes, applying optional extra processing if specified.
        """
        cube = self._combined_cube(map_item, group_cubes)
        if map_item.extra_processing is not None:
            if self.is_model_level(map_item):
                # Not so easy to add this as an extra processing step because it needs p and z.
                cube = model_level_to_pressure(cube, self.p, self.z)
            else:
                # This might be e.g. flipping the sign of some fields.
                cube = map_item.extra_processing(cube)
        return self._to_da(cube)

    def extract_da_steps(self, map_item, group_cubes):
        """Yield DataArrays for this map item, one time step at a time for model-level variables.

        Model-level variables are interpolated onto pressure levels one step at a time so that the full
        (time, pressure, lat, lon) array never exists: it is 44 GB in float64 for a 12-step N2560 file, which is
        what drove regrid tasks to ~95 GB and made them thrash on busy nodes. Everything else is yielded whole.
        """
        if not self.is_model_level(map_item):
            yield self.extract_da(map_item, group_cubes)
            return

        cube = self._combined_cube(map_item, group_cubes)
        ntime = min(cube.shape[0], self.p.shape[0])
        for i in range(ntime):
            step = model_level_to_pressure(cube, self.p, self.z, time_indices=[i], nproc=self.nproc)
            yield self._to_da(step)


class MultiMapItem:
    """Contains several MapItems and the rules for combining these (in ops)."""
    def __init__(self, items, ops, extra_processing=None, extra_attrs=None, units=None):
        self.items = items
        self.ops = ops
        self.extra_processing = extra_processing
        self.extra_attrs = extra_attrs if extra_attrs is not None else {}
        self.units = units

    def __repr__(self):
        return 'MultiMapItem(' + str(self.items) + ', ' + str(self.ops) + ')'



class MapItem:
    """Contains iris constraints to extract a given cube, along with extra_processing."""
    def __init__(self, iris_constraint, extra_processing=None, extra_attrs=None, units=None):
        self.iris_constraint = iris_constraint
        self.extra_processing = extra_processing
        self.extra_attrs = extra_attrs if extra_attrs is not None else {}
        self.units = units

    def __repr__(self):
        return f'MapItem({self.iris_constraint})'
