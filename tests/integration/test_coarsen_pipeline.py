import numpy as np
import pandas as pd
import pytest
import xarray as xr

from um_to_healpix.healpix_coarsen import _compute_coarsened

pytestmark = pytest.mark.slow

SRC_ZOOM = 3
TGT_ZOOM = 2
NCELL_SRC = 12 * 4 ** SRC_ZOOM   # 768  — must be exactly this for coarsen(4) to work
NCELL_TGT = 12 * 4 ** TGT_ZOOM   # 192
NTIMES = 4

# chunks dict keyed by zoom level, as expected by _compute_coarsened
CHUNKS = {TGT_ZOOM: (1, NCELL_TGT)}


def _make_src_ds(ntimes=NTIMES, seed=42):
    """Synthetic healpix dataset at SRC_ZOOM with valid data."""
    rng = np.random.default_rng(seed)
    time = pd.date_range('2020-01-01', periods=ntimes, freq='h')
    data = rng.random((ntimes, NCELL_SRC)).astype(np.float32)
    return xr.Dataset({
        'tas': xr.DataArray(
            data,
            dims=['time', 'healpix_index'],
            coords={'time': time, 'healpix_index': np.arange(NCELL_SRC)},
        ),
    })


@pytest.fixture(scope='module')
def coarsened_ds():
    """Run _compute_coarsened once and reuse across tests in this module."""
    return _compute_coarsened(
        _make_src_ds(),
        tgt_store=None,
        tgt_zoom=TGT_ZOOM,
        dim='2d',
        chunks=CHUNKS,
        regional=False,
        is_first_slice=True,
        has_src_weights=False,
    )


class TestComputeCoarsened:
    def test_output_healpix_cell_count(self, coarsened_ds):
        assert coarsened_ds.sizes['healpix_index'] == NCELL_TGT

    def test_output_time_size_preserved(self, coarsened_ds):
        assert coarsened_ds.sizes['time'] == NTIMES

    def test_healpix_zoom_attr_set_on_all_vars(self, coarsened_ds):
        for da in coarsened_ds.data_vars.values():
            assert da.attrs.get('healpix_zoom') == TGT_ZOOM

    def test_encoding_chunks_set(self, coarsened_ds):
        for da in coarsened_ds.data_vars.values():
            assert da.encoding.get('chunks') == CHUNKS[TGT_ZOOM]

    def test_crs_not_in_output(self, coarsened_ds):
        # _compute_coarsened always strips 'crs' from the returned dataset
        assert 'crs' not in coarsened_ds

    def test_spatial_mean_preserved(self):
        # Coarsen by mean of groups of 4 — global mean must be exactly preserved
        # (each output cell = mean of 4 inputs, so mean-of-means = overall mean).
        ds = _make_src_ds(seed=99)
        result = _compute_coarsened(
            ds, None, TGT_ZOOM, '2d', CHUNKS, False, True, False,
        )
        input_mean = float(ds['tas'].mean())
        output_mean = float(result['tas'].mean())
        np.testing.assert_allclose(output_mean, input_mean, rtol=1e-4)

    def test_output_values_in_range_of_input(self, coarsened_ds):
        src = _make_src_ds()
        src_min = float(src['tas'].min())
        src_max = float(src['tas'].max())
        out_min = float(coarsened_ds['tas'].min())
        out_max = float(coarsened_ds['tas'].max())
        assert out_min >= src_min - 1e-5
        assert out_max <= src_max + 1e-5
