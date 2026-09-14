"""Tests for um_to_healpix.checks — the store checks used by the check_store rule."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from um_to_healpix import checks


def make_ds(ncell=48, ntime=4, values=None, extra_dim=None):
    time = pd.date_range('2020-01-20', periods=ntime, freq='h')
    if extra_dim:
        name, size = extra_dim
        data = np.full((ntime, size, ncell), 280.0) if values is None else values
        ds = xr.Dataset({'tas': ((('time', name, 'cell')), data)},
                        coords={'time': time, name: np.arange(size), 'cell': np.arange(ncell)})
    else:
        data = np.full((ntime, ncell), 280.0) if values is None else values
        ds = xr.Dataset({'tas': (('time', 'cell'), data)},
                        coords={'time': time, 'cell': np.arange(ncell)})
    return ds


class TestStructure:
    def test_ok(self):
        ds = make_ds()
        assert checks.check_structure(ds, ds.time.values, 48) == []

    def test_wrong_cell_count(self):
        ds = make_ds(ncell=12)
        failures = checks.check_structure(ds, ds.time.values, 48)
        assert len(failures) == 1 and 'expected 48' in failures[0]

    def test_missing_variable(self):
        ds = make_ds()
        failures = checks.check_structure(ds, ds.time.values, 48, expected_vars=['tas', 'pr'])
        assert any("missing variables: ['pr']" in f for f in failures)

    def test_time_values_differ(self):
        ds = make_ds()
        other = pd.date_range('2021-01-01', periods=ds.sizes['time'], freq='h')
        assert checks.check_structure(ds, other, 48) == ['time coordinate values differ from the config time index']


class TestNansAndRanges:
    def test_ok(self):
        assert checks.check_nans_and_ranges(make_ds(), 0) == []

    def test_out_of_range(self):
        data = np.full((4, 48), 280.0)
        data[0, :5] = 1000.0  # tas limit is 400 K
        failures = checks.check_nans_and_ranges(make_ds(values=data), 0)
        assert len(failures) == 1 and 'above 400' in failures[0]

    def test_tiny_negative_allowed_for_non_negative_vars(self):
        ds = make_ds()
        ds['pr'] = (('time', 'cell'), np.full((4, 48), -1e-18))
        assert checks.check_nans_and_ranges(ds, 0) == []

    def test_too_many_nans(self):
        data = np.full((4, 48), 280.0)
        data[0, :40] = np.nan
        failures = checks.check_nans_and_ranges(make_ds(values=data), 0)
        assert any('NaN' in f for f in failures)

    def test_land_only_var_allowed_more_nans(self):
        """mrsol is NaN over ocean: ~68% NaN at z7 is correct, not a failure."""
        data = np.full((4, 48), 280.0)
        data[0, :33] = np.nan  # 69% NaN
        ds = make_ds(values=data).rename({'tas': 'mrsol'})
        assert checks.check_nans_and_ranges(ds, 0) == []
        # the same fraction in a global variable is still a failure
        assert len(checks.check_nans_and_ranges(make_ds(values=data), 0)) == 1

    def test_extra_dimension_checked_per_level(self):
        """mrsol is hourly but has depth: each level is checked separately."""
        data = np.full((4, 3, 48), 280.0)
        data[0, 2, :] = 1000.0
        failures = checks.check_nans_and_ranges(make_ds(values=data, extra_dim=('depth', 3)), 0)
        assert len(failures) == 1 and 'tas[depth=2]' in failures[0]


class TestLevelConsistency:
    def _pair(self, fine_values):
        fine = make_ds(ncell=48, values=fine_values)
        coarse_values = fine_values.reshape(fine_values.shape[0], 12, 4).mean(axis=-1)
        coarse = make_ds(ncell=12, values=coarse_values)
        return coarse, fine

    def test_ok(self):
        fine_values = np.random.default_rng(0).uniform(250, 300, (4, 48))
        coarse, fine = self._pair(fine_values)
        assert checks.check_level_consistency(coarse, fine, 0) == []

    def test_detects_wrong_coarsening(self):
        fine_values = np.random.default_rng(0).uniform(250, 300, (4, 48))
        coarse, fine = self._pair(fine_values)
        coarse['tas'][0, 5] += 1.0
        failures = checks.check_level_consistency(coarse, fine, 0)
        assert len(failures) == 1 and '1/12 cells differ' in failures[0]

    def test_near_zero_values_not_flagged_by_relative_error(self):
        """wa crosses zero: float32 summation order gives big *relative* differences on tiny values."""
        rng = np.random.default_rng(0)
        fine_values = rng.uniform(-10, 10, (4, 48))
        coarse, fine = self._pair(fine_values)
        coarse['tas'][0, 3] = coarse['tas'][0, 3] * 0 + 1e-7  # a near-zero cell...
        fine['tas'][0, 12:16] = 0.0                            # ...whose children mean to 0
        assert checks.check_level_consistency(coarse, fine, 0) == []

    def test_wrong_cell_ratio(self):
        coarse = make_ds(ncell=12)
        fine = make_ds(ncell=36)
        failures = checks.check_level_consistency(coarse, fine, 0)
        assert any('expected 4x12' in f for f in failures)

    def test_nan_cells_ignored(self):
        fine_values = np.random.default_rng(0).uniform(250, 300, (4, 48))
        coarse, fine = self._pair(fine_values)
        fine['tas'][0, :4] = np.nan
        coarse['tas'][0, 0] = np.nan
        assert checks.check_level_consistency(coarse, fine, 0) == []


class TestCoverage:
    @pytest.fixture
    def store(self, tmp_path):
        ds = make_ds(ncell=48, ntime=8).chunk({'time': 2})
        path = tmp_path / 'test.zarr'
        # Production stores are zarr v2 (chunk keys '0.0'); v3 nests them under 'c/'.
        ds.to_zarr(path, consolidated=True, zarr_format=2)
        return path, ds

    def test_full_coverage(self, store):
        import fsspec
        path, ds = store
        fs = fsspec.filesystem('file')
        steps = checks.written_time_steps(fs, f's3://{path}', 'tas', 8, time_chunk=2)
        assert steps == list(range(8))

    def test_end_steps_may_be_missing(self):
        """Time-means have no first step, instantaneous fields no last, and coarsening drops the final step."""
        assert checks.check_coverage_from_steps(list(range(1, 8)), 8) == []
        assert checks.check_coverage_from_steps(list(range(0, 7)), 8) == []

    def test_two_missing_at_an_end_is_a_failure(self):
        failures = checks.check_coverage_from_steps(list(range(2, 8)), 8)
        assert len(failures) == 1 and 'short by more than 1' in failures[0]

    def test_internal_gap_is_a_failure(self):
        failures = checks.check_coverage_from_steps([0, 1, 2, 5, 6, 7], 8)
        assert len(failures) == 1 and '2 missing time steps inside' in failures[0]

    def test_missing_chunk_detected(self, store):
        import fsspec
        path, ds = store
        for f in (path / 'tas').glob('3.*'):
            f.unlink()
        fs = fsspec.filesystem('file')
        steps = checks.written_time_steps(fs, f's3://{path}', 'tas', 8, time_chunk=2)
        assert steps == [0, 1, 2, 3, 4, 5]  # chunk 3 covers steps 6-7
