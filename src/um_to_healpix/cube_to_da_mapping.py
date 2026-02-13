"""Declarative mapping helpers from Iris cubes to xarray DataArrays.

This module provides lightweight container classes and an extractor that let the
pipeline define *what* to read from an Iris cube collection and *how* to
combine/post-process the result, without hard-coding those rules inline.

Core concepts
-------------
- ``MapItem``: one Iris constraint -> one cube.
- ``MultiMapItem``: multiple ``MapItem`` definitions + binary ops to combine
  extracted cubes into one field.
- ``DataArrayExtractor``: executes extraction, optional processing, and converts
  the result to an ``xarray.DataArray``.
"""
import xarray as xr

from .util import model_level_to_pressure


class DataArrayExtractor:
    """Extract and transform cubes into DataArrays using map-item metadata.

    Parameters
    ----------
    p : iris.cube.Cube
        Pressure cube used by pressure-level interpolation processing.
    z : iris.cube.Cube
        Cube that carries destination pressure coordinate values.

    Notes
    -----
    The extractor is intentionally stateful so pressure interpolation can access
    shared context (``p`` and ``z``) when requested by a map item.
    """
    def __init__(self, p, z):
        # Context cubes required for model-level -> pressure-level interpolation.
        self.p = p
        self.z = z

    @staticmethod
    def extract_cubes(map_item, group_cubes):
        """Extract one or more cubes from a grouped cube collection.

        Parameters
        ----------
        map_item : MapItem | MultiMapItem
            Declarative extraction definition.
        group_cubes : iris.cube.CubeList-like
            Input cube collection that supports ``extract_cube(constraint)``.

        Returns
        -------
        list[iris.cube.Cube]
            List containing one cube for ``MapItem`` or multiple cubes for
            ``MultiMapItem``.

        Notes
        -----
        Returning a list in both cases keeps downstream combination logic simple.
        """
        # Extract the cube and combine if necessary.
        if isinstance(map_item, MapItem):
            # Single constraint -> single extracted cube.
            cube = group_cubes.extract_cube(map_item.iris_constraint)
            return [cube]
        elif isinstance(map_item, MultiMapItem):
            # Multi-field extraction preserving declaration order in map_item.items.
            constraint_cubes = [group_cubes.extract_cube(map_item.iris_constraint) for map_item in map_item.items]
            return constraint_cubes
        else:
            # Fail fast on unsupported item type to avoid silent mis-mapping.
            raise Exception(f'unknown type {type(map_item)}')

    def extract_da(self, map_item, group_cubes):
        """Build a DataArray from map definition + grouped Iris cubes.

        Processing pipeline:
        1. Extract cube(s) using the map item constraints.
        2. Normalize known 13-step time cubes to 12 steps.
        3. If ``MultiMapItem``, combine cubes using the configured ops.
        4. Apply optional extra processing (callable or interpolation token).
        5. Convert to xarray and enforce stable variable naming.

        Parameters
        ----------
        map_item : MapItem | MultiMapItem
            Mapping rule describing extraction and optional processing.
        group_cubes : iris.cube.CubeList-like
            Source cube set for extraction.

        Returns
        -------
        xarray.DataArray
            Final mapped field ready for dataset assembly.
        """
        # *Always* shorten cubes of time 13 to length 12 by ignoring first value.
        # This applies to the first cube of each day, for certain fields.
        cubes = self.extract_cubes(map_item, group_cubes)

        # Normalize a known edge-case in input files where a leading extra step
        # appears for some variables in the first cube of a day.
        for i in range(len(cubes)):
            cube = cubes[i]
            if cube.shape[0] == 13:
                cubes[i] = cube[1:]

        # Start from first extracted cube; combine subsequent cubes if configured.
        cube = cubes[0]
        if isinstance(map_item, MultiMapItem) and len(cubes) > 1:
            # Apply each binary operator in sequence: cube = op(cube, next_cube).
            for next_cube, op in zip(cubes[1:], map_item.ops):
                cube.data = op(cube.data, next_cube.data)

        # Apply optional post-processing step.
        if map_item.extra_processing is not None:
            if map_item.extra_processing == 'interpolate_model_levels_to_pressure':
                # Not so easy to add this as an extra processing step because it needs p and z.
                # Delegate to utility that performs vertical interpolation.
                cube = model_level_to_pressure(cube, self.p, self.z)
            else:
                # This might be e.g. flipping the sign of some fields.
                # For callable processors, pass/return Iris cube.
                cube = map_item.extra_processing(cube)

        # For some cubes (ones with names like m01s30i461 the da gets a name like filled-XXXXXX...
        # Make sure it's got the actual cube name so I can rename it later.
        # Enforce stable DataArray name from Iris cube metadata.
        da = xr.DataArray.from_iris(cube).rename(cube.name())
        return da


class MultiMapItem:
    """Bundle multiple map items plus rules to combine extracted cubes.

    Parameters
    ----------
    items : list[MapItem]
        Ordered map item definitions to extract.
    ops : list[callable]
        Binary operations applied left-to-right between consecutive cube data
        arrays. Typically ``len(ops) == len(items) - 1``.
    extra_processing : callable | str | None, optional
        Optional post-combination processing. Special string token
        ``'interpolate_model_levels_to_pressure'`` triggers interpolation path
        requiring extractor context.
    extra_attrs : dict | None, optional
        Extra metadata to attach downstream when constructing datasets.
    """
    def __init__(self, items, ops, extra_processing=None, extra_attrs=None):
        # Ordered extraction definitions.
        self.items = items
        # Ordered pairwise operations used during combination.
        self.ops = ops
        # Optional post-processing descriptor/callable.
        self.extra_processing = extra_processing
        # Optional additional attributes; default empty dict for convenience.
        self.extra_attrs = extra_attrs if extra_attrs is not None else {}

    def __repr__(self):
        # Compact developer-friendly representation for logs/debugging.
        return 'MutliMapItem(' + str(self.items) + ', ' + str(self.ops) + ')'



class MapItem:
    """Declarative definition for extracting and optionally post-processing one cube.

    Parameters
    ----------
    iris_constraint : iris.Constraint
        Constraint used with ``extract_cube``.
    extra_processing : callable | str | None, optional
        Optional post-processing step. May be a callable taking/returning an
        Iris cube, or a special token interpreted by ``DataArrayExtractor``.
    extra_attrs : dict | None, optional
        Extra metadata carried alongside this mapping for downstream use.
    """
    def __init__(self, iris_constraint, extra_processing=None, extra_attrs=None):
        # Primary extraction rule.
        self.iris_constraint = iris_constraint
        # Optional post-extraction transformation.
        self.extra_processing = extra_processing
        # Optional metadata augmentation.
        self.extra_attrs = extra_attrs if extra_attrs is not None else {}

    def __repr__(self):
        # Helpful concise representation in debugging output.
        return f'MapItem({self.iris_constraint})'
