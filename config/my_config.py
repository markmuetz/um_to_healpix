"""End-to-end processing configuration for the HK26 UM -> HEALPix pipeline.

This module defines one top-level dictionary, ``processing_config``, keyed by
simulation identifier. Each simulation config contains everything needed by the
controller/worker CLIs to:

1. locate source UM ``.pp`` files,
2. choose variable mappings and cube constraints,
3. control chunking and target Zarr layout,
4. define done-file paths for incremental reruns,
5. route output to the correct object-store destination.

Operational usage (via CLI) typically includes:
        um-slurm-control -C config/hk26_config.py ls
        um-slurm-control -C config/hk26_config.py print-config <simulation-key>

Design notes:
- This file is intentionally Python (not YAML) so config sections can be reused
    via variables/copies and lightly customized per simulation family.
- Comments here are extensive by design to make operational behavior explicit.
"""
import copy
import operator
from pathlib import Path

import iris.cube
import pandas as pd

from um_to_healpix.cube_to_da_mapping import MapItem, MultiMapItem
from um_to_healpix.util import has_dimensions, cube_cell_method_is_not_empty, cube_cell_method_is_empty, \
    invert_cube_sign, check_cube_time_length

# ---------------------------------------------------------------------------
# GLOBAL SWITCHES AND ROOT PATHS
# ---------------------------------------------------------------------------
# Output version used in:
# - done-file namespaces
# - output object-store paths
# Bumping this creates a new processing lineage.
output_vn = 'v6.2'

# Deployment namespace used in object-store and done-file locations.
# Common examples are 'dev' or 'prod'-like environments.
deploy = 'dev'

# Location of input files.
dy3dir = Path('/gws/nopw/j04/kscale/DYAMOND3_reruns/')

# Location where HEALPix remapping weights are stored/reused.
weightsdir = Path('/gws/nopw/j04/hrcm/sharar/weights/')

# Location of donefiles. Delete to rerun a particular task.
donedir = Path(f'/gws/nopw/j04/hrcm/sharar/slurm_done/{deploy}')

# ---------------------------------------------------------------------------
# DEFAULT SLURM SETTINGS
# ---------------------------------------------------------------------------
# These defaults are consumed by orchestration logic and can be overridden
# externally when dispatching specific jobs/workloads.
slurm_config = dict(
    account='hrcm',
    ntasks=1,
    cpus_per_task=1,
    partition='standard',
    qos='standard',
    time='24:00:00',
    mem=100000,
    # Concurrency cap to avoid overwhelming shared object-store services.
    nconcurrent_tasks=40,
)

# ---------------------------------------------------------------------------
# SHARED DATASET METADATA
# ---------------------------------------------------------------------------
# Metadata added to outputs across simulations to provide scientific context.
shared_metadata = {
    'Met Office DYAMOND3 simulations': (
        'A group of experiments have been conducted using the Met Office Unified Model (MetUM) with a focus on the '
        'DYAMOND-3 period (Jan 2020-Feb 2021). While this experiments include standalone explicit convection global '
        'simulations we have also developed a cyclic tropical channel and include limited area model simulations to '
        'build our understanding of how resolving smaller-scale processes feeds back on to the large-scale '
        'atmospheric circulation.'),
}


# ---------------------------------------------------------------------------
# CHUNKING STRATEGY
# ---------------------------------------------------------------------------
# Chunk dictionaries are indexed by HEALPix zoom level and map to
# (time_chunk, spatial_chunk) for 2D arrays, and (time, level, spatial) for 3D.
#
# Guiding principle: target roughly 1-10MB chunk sizes while balancing
# write amplification and downstream read efficiency.
chunks2d = {
    # z10 and z9 have to have no chunking over time. (z9 is the highest zoom for N1280.)
    10: (1, 4 ** 10),  # 12 chunks per time.
    9: (1, 4 ** 9),
    # Increase temporal at same rate as reducing spatial.
    8: (4 ** 2, 4 ** 8),
    7: (4 ** 3, 4 ** 7),
    # Transition to 3 chunks per time.
    6: (4 ** 3, 4 ** 7),
    # Transition to 1 chunk per time.
    5: (4 ** 3, 12 * 4 ** 5),
    4: (4 ** 4, 12 * 4 ** 4),
    3: (4 ** 5, 12 * 4 ** 3),
    2: (4 ** 6, 12 * 4 ** 2),
    1: (4 ** 7, 12 * 4 ** 1),
    0: (4 ** 8, 12 * 4 ** 0),
}

# 3D global chunks reuse the same spatial chunk as 2D, with level chunk = 1.
# This generally performs better for vertical slicing and regional operations.
chunks3d = {
    z: (t, 1, s)
    for z, (t, s) in chunks2d.items()
}

# Regional chunk profile differs at high zoom to keep active-region writes
# efficient and avoid excessive tiny chunks.
chunks2dregional = {
    # z10 has to have no chunking over time.
    10: (1, 4 ** 9),  # 12 x 4 = 48 chunks per time.
    # Increase temporal at same rate as reducing spatial.
    9: (4, 4 ** 9),
    8: (4 ** 2, 4 ** 8),
    7: (4 ** 3, 4 ** 7),
    # Transition to 3 chunks per time.
    6: (4 ** 3, 4 ** 7),
    # Transition to 1 chunk per time.
    5: (4 ** 3, 12 * 4 ** 5),
    4: (4 ** 4, 12 * 4 ** 4),
    3: (4 ** 5, 12 * 4 ** 3),
    2: (4 ** 6, 12 * 4 ** 2),
    1: (4 ** 7, 12 * 4 ** 1),
    0: (4 ** 8, 12 * 4 ** 0),
}

# Regional 3D chunks mirror regional 2D spatial settings.
chunks3dregional = {
    z: (t, 1, s)
    for z, (t, s) in chunks2dregional.items()
}

# ---------------------------------------------------------------------------
# VARIABLES TO DROP
# ---------------------------------------------------------------------------
# Coordinate/auxiliary fields in source cubes that are not needed in output
# stores and may cause conflicts or unnecessary payload inflation.
drop_vars = [
    'latitude_0',
    'longitude_0',
    # 'bnds',  # gets dropped automatically and causes problems if you try to drop it.
    'forecast_period',
    'forecast_reference_time',
    'forecast_period_0',
    'height',
    'height_0',
    'forecast_period_1',
    'forecast_period_2',
    'forecast_period_3',
    'latitude_longitude',
    'time_1_bnds',
    'time_0_bnds',
    'forecast_period_1_bnds',
    'surface_altitude',
    'level_height',
    'sigma',
    'altitude',
]

# ---------------------------------------------------------------------------
# TIME AXIS TEMPLATES
# ---------------------------------------------------------------------------
# Canonical target timelines used for indexing/writing.
# - 2D variables are hourly.
# - 3D variables are 3-hourly.
time2d = pd.date_range('2020-01-20', '2021-03-01', freq='h')
time3d = pd.date_range('2020-01-20', '2021-03-01', freq='3h')

# ---------------------------------------------------------------------------
# VARIABLE MAPPINGS: SOURCE CUBES -> OUTPUT VARIABLES
# ---------------------------------------------------------------------------
# Mapping schema:
#   key = (<short_name>, <CF-like long_name>)
#   val = MapItem(...) or MultiMapItem(...)
#
# MapItem:
#   - selects a source cube via Iris constraint (or name string),
#   - may apply extra processing (e.g., sign flips, level interpolation),
#   - may add per-variable attrs (e.g., notes).
#
# MultiMapItem:
#   - combines multiple MapItems with operators (operator.add here),
#   - used for merged diagnostics such as total precipitation.
name_map_2d = {
    ('psl', 'air_pressure_at_mean_sea_level'): MapItem('air_pressure_at_sea_level'),
    ('tas', 'air_temperature'): MapItem('air_temperature'),
    ('psl', 'air_pressure_at_mean_sea_level'): MapItem('air_pressure_at_sea_level'),
    ('tas', 'air_temperature'): MapItem('air_temperature'),
    # TODO: Why are these two missing? Can I replace them?
    # ('clwvi', 'atmosphere_mass_content_of_cloud_condensed_water'): MapItem('atmosphere_cloud_liquid_water_content'),
    # ('clivi', 'atmosphere_mass_content_of_cloud_ice'): MapItem('atmosphere_cloud_ice_content'),
    ('prw', 'atmosphere_mass_content_of_water_vapor'): MapItem('m01s30i461'),
    ('clt', 'cloud_area_fraction'): MapItem('cloud_area_fraction_assuming_maximum_random_overlap'),
    ('uas', 'eastward_wind'): MapItem('x_wind'),
    ('vas', 'northward_wind'): MapItem('y_wind'),
    ('pr', 'precipitation_flux'): MultiMapItem(
        [MapItem(iris.Constraint(name='stratiform_rainfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty)),
         MapItem(iris.Constraint(name='stratiform_snowfall_flux') & iris.Constraint(
             cube_func=cube_cell_method_is_not_empty))],
        ops=[operator.add],
        extra_attrs={
            'notes': 'Combined rain and snow. Hourly mean - time index shifted from half past the hour to the following hour'},
    ),
    ('prs', 'solid_precipitation_flux'): MapItem(
        iris.Constraint(name='stratiform_snowfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty),
        extra_attrs={'notes': 'Hourly mean - time index shifted from half past the hour to the following hour'},
        ),
    ('huss', 'specific_humidity'): MapItem('specific_humidity'),
    ('ps', 'surface_air_pressure'): MapItem('surface_air_pressure'),
    ('hflsd', 'surface_downward_latent_heat_flux'): MapItem('surface_upward_latent_heat_flux',
                                                            extra_processing=invert_cube_sign),
    ('hfssd', 'surface_downward_sensible_heat_flux'): MapItem('surface_upward_sensible_heat_flux',
                                                              extra_processing=invert_cube_sign),
    ('rlds', 'surface_downwelling_longwave_flux_in_air'): MapItem('surface_downwelling_longwave_flux_in_air'),
    ('rldscs', 'surface_downwelling_longwave_flux_in_air_clear_sky'): MapItem(
        'surface_downwelling_longwave_flux_in_air_assuming_clear_sky'),
    ('rsds', 'surface_downwelling_shortwave_flux_in_air'): MapItem('surface_downwelling_shortwave_flux_in_air'),
    ('rsdscs', 'surface_downwelling_shortwave_flux_in_air_clear_sky'): MapItem(
        'surface_downwelling_shortwave_flux_in_air_assuming_clear_sky'),
    ('ts', 'surface_temperature'): MapItem('surface_temperature'),
    ('rsdt', 'toa_incoming_shortwave_flux'): MapItem('toa_incoming_shortwave_flux'),
    ('rlut', 'toa_outgoing_longwave_flux'): MapItem('toa_outgoing_longwave_flux'),
    ('rlutcs', 'toa_outgoing_longwave_flux_clear_sky'): MapItem('toa_outgoing_longwave_flux_assuming_clear_sky'),
    ('rsut', 'toa_outgoing_shortwave_flux'): MapItem(iris.Constraint(
        name='toa_outgoing_shortwave_flux') & iris.AttributeConstraint(
        STASH='m01s01i208')),
    ('rsutcs', 'toa_outgoing_shortwave_flux_clear_sky'): MapItem('toa_outgoing_shortwave_flux_assuming_clear_sky'),
    ('rsus', 'surface_upwelling_shortwave_flux_in_air'): MapItem('m01s01i202', extra_processing=invert_cube_sign),
    ('rlus', 'surface_upwelling_longwave_flux_in_air'): MapItem('surface_net_downward_longwave_flux', extra_processing=invert_cube_sign),
}

# This variable has a depth coordinate but is PT1H frequency, so operationally
# it is grouped with 2D hourly products in a dedicated group.
name_map_2d_depth = {
    ('mrso', 'soil_liquid_water_content'): MapItem('moisture_content_of_soil_layer'),
}

name_map_3d = {
    ('ua', 'eastward_wind'): MapItem('x_wind'),
    ('zg', 'geopotential height'): MapItem('geopotential_height'),
    ('va', 'northtward_wind'): MapItem('y_wind'),
    ('hur', 'relative_humidity'): MapItem(iris.Constraint(
        name='relative_humidity') & iris.AttributeConstraint(
        STASH='m01s30i206')),
    ('hus', 'specific_humidity'): MapItem('specific_humidity'),
    ('ta', 'temperature'): MapItem('air_temperature'),
    ('wa', 'upward_air_velocity'): MapItem('upward_air_velocity'),
}

name_map_3d_ml = {
    ('cli', 'mass_fraction_of_cloud_ice_in_air'): MapItem('mass_fraction_of_cloud_ice_in_air',
                                                          extra_processing='interpolate_model_levels_to_pressure'),
    ('clw', 'mass_fraction_of_cloud_liquid_water_in_air'): MapItem('mass_fraction_of_cloud_liquid_water_in_air',
                                                                   extra_processing='interpolate_model_levels_to_pressure'),
    ('qg', 'mass_fraction_of_graupel_in_air'): MapItem('mass_fraction_of_graupel_in_air',
                                                       extra_processing='interpolate_model_levels_to_pressure'),
    ('qr', 'mass_fraction_of_rain_in_air'): MapItem('mass_fraction_of_rain_in_air',
                                                    extra_processing='interpolate_model_levels_to_pressure'),
    ('qs', 'mass_fraction_of_snow_water_in_air'): MapItem('mass_fraction_of_cloud_ice_crystals_in_air',
                                                          extra_processing='interpolate_model_levels_to_pressure'),
}

# ---------------------------------------------------------------------------
# VARIABLE GROUPS
# ---------------------------------------------------------------------------
# Groups define a coherent processing bucket:
# - target time index,
# - destination store frequency (PT1H/PT3H),
# - variable mapping table,
# - dimensional constraint used to subset candidate cubes,
# - chunking profile by zoom.
#
# The 'constraint' key is important when cube names overlap across dimensions;
# it prevents accidental cross-group selection.
group2d = {
    'time': time2d,
    'zarr_store': 'PT1H',
    'name_map': name_map_2d,
    'constraint': has_dimensions("time", "latitude", "longitude"),
    'chunks': chunks2d,
}

group2d_depth = {
    'time': time2d,
    'zarr_store': 'PT1H',
    'name_map': name_map_2d_depth,
    'constraint': has_dimensions("time", "depth", "latitude", "longitude"),
    'chunks': chunks3d,
}

group3d = {
    'time': time3d,
    'zarr_store': 'PT3H',
    'name_map': name_map_3d,
    'constraint': has_dimensions("time", "pressure", "latitude", "longitude"),
    'chunks': chunks3d,
}

group3d_ml = {
    'time': time3d,
    # Note, uses same zarr_store as group3d so ends up in same 3D store.
    'zarr_store': 'PT3H',
    'name_map': name_map_3d_ml,
    'constraint': has_dimensions("time", "model_level_number", "latitude", "longitude"),
    'chunks': chunks3d,
}

# ---------------------------------------------------------------------------
# SIMULATION KEY REGISTRIES
# ---------------------------------------------------------------------------
# These registries map canonical simulation keys to source directory prefixes.
# Key naming convention is also used downstream in done paths and output URLs.
#
# Idea is to map to e.g. these filenames.
# Global:
# ./10km-CoMA9/glm/field.pp/apvera.pp/glm.n1280_CoMA9.apvera_20200120T00.pp
# ./5km-RAL3/glm/field.pp/apvera.pp/glm.n2560_RAL3p3.apvera_20200120T00.pp
# ./10km-GAL9-nest/glm/field.pp/apvera.pp/glm.n1280_GAL9_nest.apvera_20200120T00.pp
global_sim_keys = {
    'glm.n2560_RAL3p3.tuned': '5km-RAL3p3-tuned',
    'glm.n1280_CoMA9': '10km-CoMA9',
    # 'glm.n1280_GAL9_nest': '10km-GAL9-nest',
}

# ---------------------------------------------------------------------------
# SIMULATION-FAMILY SPECIAL CASES
# ---------------------------------------------------------------------------
# Different simulation families encode precipitation differently.
# The base 2D group is cloned and patched per-family to keep behavior explicit
# while minimizing duplication.

# For GAL9 simulations, combine stratiform and convective, rainfall and snow.
group2d_GAL9 = copy.deepcopy(group2d)
group2d_GAL9['name_map'][('pr', 'precipitation_flux')] = MultiMapItem(
    [
        MapItem(iris.Constraint(name='convective_rainfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty)),
        MapItem(iris.Constraint(name='convective_snowfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty)),
        MapItem(iris.Constraint(name='stratiform_rainfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty)),
        MapItem(iris.Constraint(name='stratiform_snowfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty)),
    ],
    ops=[operator.add, operator.add, operator.add],
    extra_attrs={
        'notes': 'Combined rain and snow, convective and stratiform. Hourly mean - time index shifted from half past the hour to the following hour'},
)

# For CoMA9 simulations, use instantaneous total precipitation diagnostic.
# check_cube_time_length handles known first-file edge cases.
group2d_CoMA9 = copy.deepcopy(group2d)
group2d_CoMA9['name_map'][('pr', 'precipitation_flux')] = MapItem(
    iris.AttributeConstraint(STASH='m01s05i216') & iris.Constraint(cube_func=cube_cell_method_is_empty),
    extra_processing=check_cube_time_length,
    extra_attrs={
        'notes': 'Uses instantaneous total precipitation'},
    )

group3d_ml_CoMA9 = copy.deepcopy(group3d_ml)
# TODO: Not present.
del group3d_ml_CoMA9['name_map'][('qs', 'mass_fraction_of_snow_water_in_air')]

# Build simulation-key -> group mapping so each simulation gets the correct
# family-specific variable behavior.
group2d_global_map = {}
for key in global_sim_keys:
    if key.endswith('GAL9_nest'):
        group2d_global_map[key] = group2d_GAL9
    elif key.endswith('CoMA9'):
        group2d_global_map[key] = group2d_CoMA9
    else:
        group2d_global_map[key] = group2d

group3d_ml_global_map = {}
for key in global_sim_keys:
    if key.endswith('GAL9_nest'):
        group3d_ml_global_map[key] = group3d_ml
    elif key.endswith('CoMA9'):
        group3d_ml_global_map[key] = group3d_ml_CoMA9
    else:
        group3d_ml_global_map[key] = group3d_ml

# ---------------------------------------------------------------------------
# GLOBAL SIMULATION CONFIG CONSTRUCTION
# ---------------------------------------------------------------------------
# Each config entry defines all required runtime fields expected by the
# processing engine.
global_configs = {
    key: {
        # Canonical simulation identifier.
        'name': key,
        # Distinguishes global from limited-area/regional logic branches.
        'regional': False,
        # Enables cyclic extension when required by global remapping.
        'add_cyclic': True,
        # Source root containing simulation PP files.
        'basedir': dy3dir / f'{simdir}/glm',
        # Weight cache root for remapping products.
        'weightsdir': weightsdir,
        # Done-file root for incremental task completion tracking.
        'donedir': donedir,
        # Per-task done-file template. {task}/{date} are populated at runtime.
        'donepath_tpl': f'{key}/{output_vn}/{{task}}_{{date}}.done',
        # Lower temporal bound for task generation.
        'first_date': pd.Timestamp(2020, 1, 20, 0),
        # Highest HEALPix zoom differs by native simulation resolution.
        'max_zoom': 10 if key.startswith('glm.n2560') else 9,
        # Object-store output template. freq/zoom are expanded per group/task.
        'zarr_store_url_tpl': f's3://sim-data/{deploy}/{output_vn}/{key}/um.{{freq}}.hp_z{{zoom}}.zarr',
        # Vars removed before writing to reduce noise/conflicts.
        'drop_vars': drop_vars,
        # Processing groups enabled for this simulation.
        'groups': {
            '2d': group2d_global_map[key],
            '2d_depth': group2d_depth,
            '3d': group3d,
            '3d_ml': group3d_ml_global_map[key],
        },
        # Metadata persisted into outputs.
        'metadata': {
            'simulation': key,
        }
    }
    for key, simdir in global_sim_keys.items()
}

global_configs['glm.n2560_RAL3p3.tuned']['metadata'].update({
    'simulation_description': ('The MetUM uses a regular lat-lon grid, for our explicit convection global simulations '
                               'we use the N2560 global grid (~5 km in mid-latitudes) and the latest regional '
                               'atmosphere-land configuration (RAL3p3). As detailed in Bush et al 2025 the RAL3p3 '
                               'includes significant developments over previous configurations including the CASIM '
                               'double-moment cloud microphysics scheme and the bi-modal large-scale cloud scheme. '
                               'Crucially for DYAMOND-3 simulations the parameterisation of convection is not active '
                               'and this science configuration has been developed and evaluated targetting '
                               'high-resolution (regional) simulations.'),
})

# ---------------------------------------------------------------------------
# REGIONAL SIMULATION REGISTRY AND CONFIG CONSTRUCTION
# ---------------------------------------------------------------------------
# Regional:
# ./10km-GAL9-nest/Africa_km4p4_CoMA9_TBv1/field.pp/apvera.pp/Africa_km4p4_CoMA9_TBv1.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/Africa_km4p4_RAL3P3/field.pp/apvera.pp/Africa_km4p4_RAL3P3.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/SAmer_km4p4_CoMA9_TBv1/field.pp/apvera.pp/SAmer_km4p4_CoMA9_TBv1.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/SAmer_km4p4_RAL3P3/field.pp/apvera.pp/SAmer_km4p4_RAL3P3.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/SEA_km4p4_CoMA9_TBv1/field.pp/apvera.pp/SEA_km4p4_CoMA9_TBv1.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/SEA_km4p4_RAL3P3/field.pp/apvera.pp/SEA_km4p4_RAL3P3.n1280_GAL9_nest.apvera_20200120T00.pp
# Cyclic Tropical Channel (CTC):
# ./10km-GAL9-nest/CTC_km4p4_CoMA9_TBv1/field.pp/apvera.pp/CTC_km4p4_CoMA9_TBv1.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/CTC_km4p4_RAL3P3/field.pp/apvera.pp/CTC_km4p4_RAL3P3.n1280_GAL9_nest.apvera_20200120T00.pp
def map_regional_key_to_path(simdir, regional_key):
    # Regional filenames include a prefix before the dotted experiment key.
    # This helper normalizes key->path mapping for filesystem layout.
    sim_key, _ = regional_key.split('.')
    return Path(f'/gws/nopw/j04/kscale/DYAMOND3_data/{simdir}/{sim_key}')

regional_sim_keys = {
    'SAmer_km4p4_RAL3P3.n1280_GAL9_nest': '10km-GAL9-nest',
    'Africa_km4p4_RAL3P3.n1280_GAL9_nest': '10km-GAL9-nest',
    'SEA_km4p4_RAL3P3.n1280_GAL9_nest': '10km-GAL9-nest',
    'SAmer_km4p4_CoMA9_TBv1.n1280_GAL9_nest': '10km-GAL9-nest',
    'Africa_km4p4_CoMA9_TBv1.n1280_GAL9_nest': '10km-GAL9-nest',
    'SEA_km4p4_CoMA9_TBv1.n1280_GAL9_nest': '10km-GAL9-nest',
    'CTC_km4p4_RAL3P3.n1280_GAL9_nest': '10km-GAL9-nest',
    'CTC_km4p4_CoMA9_TBv1.n1280_GAL9_nest': '10km-GAL9-nest',
}

group2d_regional = copy.deepcopy(group2d)
group3d_regional = copy.deepcopy(group3d)
group3d_ml_regional = copy.deepcopy(group3d_ml)

# Regional configurations use the regional chunk profile for better
# active-domain IO characteristics.
group2d_regional['chunks'] = chunks2dregional
group3d_regional['chunks'] = chunks3dregional
group3d_ml_regional['chunks'] = chunks3dregional

regional_configs = {
    key: {
        # Flags this entry for regional processing path.
        'regional': True,
        # TODO: I think that the CTC simulation has a high enough res that there are no healpix coords outside
        # its domain - check.
        # Orig: I think this should be true for CTC, but it's raising an error: ValueError: The coordinate must be equally spaced.
        # 'add_cyclic': key.startswith('CTC'),  # only difference from regional.
        'add_cyclic': False,
        # Source location for this regional simulation.
        'basedir': dy3dir / f'{simdir}/glm',
        'weightsdir': weightsdir,
        'donedir': donedir,
        'donepath_tpl': f'{key}/{output_vn}/{{task}}_{{date}}.done',
        # Regional runs in this config are all processed to z10.
        'max_zoom': 10,
        'first_date': pd.Timestamp(2020, 1, 20, 0),
        'zarr_store_url_tpl': f's3://sim-data/{deploy}/{output_vn}/{key}/um.{{freq}}.hp_z{{zoom}}.zarr',
        'drop_vars': drop_vars,
        'groups': {
            '2d': group2d_regional,
            '3d': group3d_regional,
            '3d_ml': group3d_ml_regional,
        },
        'metadata': {
            'simulation': key,
        }
    }
    for key, simdir in regional_sim_keys.items()
}

# ---------------------------------------------------------------------------
# FINAL EXPORTED CONFIG
# ---------------------------------------------------------------------------
# processing_config is the object consumed by orchestration tools.
#
# Regional configs are currently disabled here (commented) by design; enable
# when regional runs are intended in this deployment.
processing_config = {
    **global_configs,
    # **regional_configs,
}