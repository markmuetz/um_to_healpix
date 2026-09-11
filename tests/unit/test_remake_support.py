"""Tests for behaviour needed by the remake3 remakefiles (remakefile_regrid.py/remakefile_coarsen.py)."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from um_to_healpix import um_process_tasks
from um_to_healpix.um_process_tasks import UMProcessTasks
from um_to_healpix.um_slurm_control import find_dyamond3_pp_dates_to_paths

STREAMS = ['apvera', 'apverb', 'apverc', 'apverd']


def _touch_pp(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')


def _make_pp_tree(basedir, subdir_tpl, dates, streams=STREAMS):
    for stream in streams:
        for date in dates:
            _touch_pp(basedir / subdir_tpl.format(stream=stream) / f'glm.sim.{stream}_{date}.pp')


class TestFindPPDatesToPaths:
    def test_default_glob_field_pp_layout(self, tmp_path):
        _make_pp_tree(tmp_path, 'field.pp/{stream}.pp', ['20200120T0000Z', '20200120T1200Z'])
        dates_to_paths = find_dyamond3_pp_dates_to_paths(tmp_path)
        assert sorted(dates_to_paths) == [pd.Timestamp('2020-01-20 00:00'), pd.Timestamp('2020-01-20 12:00')]
        assert all(len(v) == 4 for v in dates_to_paths.values())

    def test_default_glob_misses_flat_layout(self, tmp_path):
        _make_pp_tree(tmp_path, '{stream}', ['20200120T0000Z'])
        assert find_dyamond3_pp_dates_to_paths(tmp_path) == {}

    def test_custom_glob_flat_layout(self, tmp_path):
        _make_pp_tree(tmp_path, '{stream}', ['20200120T0000Z', '20200120T1200Z'])
        # apvere files are skipped, incomplete dates (<4 streams) are dropped.
        _make_pp_tree(tmp_path, '{stream}', ['20200120T0000Z'], streams=['apvere'])
        _make_pp_tree(tmp_path, '{stream}', ['20200121T0000Z'], streams=STREAMS[:3])
        dates_to_paths = find_dyamond3_pp_dates_to_paths(tmp_path, 'apve*/*.pp')
        assert sorted(dates_to_paths) == [pd.Timestamp('2020-01-20 00:00'), pd.Timestamp('2020-01-20 12:00')]
        assert all(len(v) == 4 for v in dates_to_paths.values())
        assert not any('apvere' in str(p) for v in dates_to_paths.values() for p in v)


class TestCoarsenHealpixRegionDonepath:
    def _run(self, monkeypatch, tgt_times):
        coarsen = MagicMock()
        monkeypatch.setattr(um_process_tasks, 'coarsen_healpix_zarr_region', coarsen)
        monkeypatch.setattr(um_process_tasks.xr, 'open_zarr', MagicMock())
        monkeypatch.setattr(um_process_tasks, 'LocalCluster', MagicMock())
        config = {
            'drop_vars': [],
            'groups': {'2d': {'chunks': {9: (1, 4 ** 9)}}},
            'zarr_store_url_tpl': 's3://bucket/um.{freq}.hp_z{zoom}.zarr',
            'regional': False,
        }
        proc = UMProcessTasks(config, {}, store_factory=lambda url: url)
        proc.coarsen_healpix_region({'dim': '2d', 'tgt_zoom': 9, 'tgt_times': tgt_times})
        return coarsen

    def test_no_donepath(self, monkeypatch):
        coarsen = self._run(monkeypatch, [{'start_idx': 0, 'end_idx': 1}, {'start_idx': 1, 'end_idx': 2}])
        assert coarsen.call_count == 2

    def test_donepath_written(self, monkeypatch, tmp_path):
        donepath = tmp_path / 'coarsen' / '0.done'
        coarsen = self._run(monkeypatch, [{'start_idx': 0, 'end_idx': 1, 'donepath': str(donepath)}])
        assert coarsen.call_count == 1
        assert donepath.exists()


class TestCreateStoresGuard:
    def _proc(self, tmp_path):
        config = {
            'drop_vars': [],
            'max_zoom': 1,
            'groups': {'2d': {'zarr_store': 'PT1H'}, '3d': {'zarr_store': 'PT3H'}},
            'zarr_store_url_tpl': str(tmp_path / 'um.{freq}.hp_z{zoom}.zarr'),
        }
        return UMProcessTasks(config, {}, store_factory=lambda url: url)

    def test_no_existing_stores(self, tmp_path):
        assert self._proc(tmp_path)._existing_store_urls() == []

    def test_refuses_to_overwrite_existing_store(self, tmp_path):
        store = tmp_path / 'um.PT3H.hp_z0.zarr'
        store.mkdir()
        (store / '.zmetadata').write_text('{}')
        proc = self._proc(tmp_path)
        assert proc._existing_store_urls() == [str(store)]
        with pytest.raises(FileExistsError, match='Refusing to recreate'):
            proc.create_empty_zarr_stores({'inpaths': []})


def test_task_log_writes_and_detaches(tmp_path):
    from loguru import logger
    from um_to_healpix.util import task_log

    path = tmp_path / 'logs' / 'task.log'
    with task_log(path):
        logger.info('inside')
    logger.info('outside')
    text = path.read_text()
    assert 'inside' in text and 'outside' not in text
