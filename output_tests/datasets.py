ZOOMS = list(range(0, 11))  # 0–10 inclusive
FREQS = ['PT1H', 'PT3H']

HK26_GLOBAL_SIMS = [
    'um_glm_n1280_CoMA9_hk26',
    'um_glm_n1280_GAL9_v2_hk26',
    'um_glm_n2560_CoMA9_hk26',
    'um_glm_n2560_RAL3p3_tuned_hk26',
]

HK26_LAM_SIMS = [
    'um_Africa_km4p4_CoMA9_TBv1_n2560_CoMA9_hier_v2_hk26',
    'um_Africa_km4p4_RAL3P3_n2560_CoMA9_nest_hk26',
    'um_CTC_km4p4_CoMA9_TBv1_n2560_CoMA9_hier_v2_hk26',
    'um_CTC_km4p4_RAL3P3_n2560_CoMA9_nest_hk26',
    'um_SAmer_km4p4_CoMA9_TBv1_n2560_CoMA9_hier_v2_hk26',
    'um_SAmer_km4p4_RAL3P3_n2560_CoMA9_nest_hk26',
    'um_SEA_km4p4_CoMA9_TBv1_n2560_CoMA9_hier_v2_hk26',
    'um_SEA_km4p4_RAL3P3_n2560_CoMA9_nest_hk26',
]

HK26_SIMS = HK26_GLOBAL_SIMS + HK26_LAM_SIMS

# Physically plausible ranges (vmin, vmax); None means unbounded.
RANGE_CHECKS = {
    'tas':  (150, 400),   # K
    'pr':   (0, None),    # kg m-2 s-1, non-negative
    'rlut': (0, None),    # W m-2, non-negative
    'clw':  (0, None),    # kg kg-1, non-negative
}

# Absolute tolerance for range checks — permits tiny negative values from
# floating-point rounding during regridding (observed values ~1e-18).
RANGE_CHECK_ATOL = 1e-10

# Global sims should have < this fraction of NaNs per variable at any zoom
MAX_NAN_FRACTION_GLOBAL = 0.50

# LAM sims are not expected to have valid data at zoom <= this value
LAM_MIN_ZOOM = 2
