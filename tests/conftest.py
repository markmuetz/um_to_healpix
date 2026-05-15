import iris
import pandas as pd

import pytest

iris.FUTURE.date_microseconds = True


def make_test_config(tmp_path, *, zoom=2, regional=False, **overrides):
    """Return a minimal processing config dict suitable for tests.

    The zarr_store_url_tpl uses local paths so a local_store_factory
    (i.e. ``lambda url: url``) can be used instead of S3.
    """
    from um_to_healpix.util import has_dimensions

    time2d = pd.date_range('2020-01-20', periods=4, freq='h')

    base = {
        'name': 'test_config',
        'regional': regional,
        'add_cyclic': True,
        'weightsdir': tmp_path / 'weights',
        'donedir': tmp_path / 'done',
        'donepath_tpl': 'test/{task}_{date}.done',
        'coarsen_donepath_tpl': 'test/coarsen/{dim}/z{zoom}/{job_id}.done',
        'first_date': pd.Timestamp('2020-01-20'),
        'max_zoom': zoom,
        'zarr_store_url_tpl': str(tmp_path / '{freq}_z{zoom}.zarr'),
        'drop_vars': [],
        'groups': {
            '2d': {
                'time': time2d,
                'zarr_store': 'PT1H',
                'name_map': {},
                'constraint': has_dimensions('time', 'latitude', 'longitude'),
                'chunks': {z: (1, 12 * 4 ** z) for z in range(zoom + 1)},
            },
        },
        'metadata': {'simulation': 'test'},
    }
    return {**base, **overrides}


def local_store_factory(url):
    """Store factory for tests: returns the URL as-is (zarr accepts local paths)."""
    return url
