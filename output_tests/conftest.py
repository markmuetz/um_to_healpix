import matplotlib
matplotlib.use('Agg')

from pathlib import Path

import pytest
import intake
import easygems.healpix as egh


def hp_mods(ds):
    return ds.rename({'healpix_index': 'cell'}).pipe(egh.attach_coords)


DEFAULT_CATALOG_URL = 'https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml'
DEFAULT_PLOT_DIR = Path(__file__).parent / 'figures'


def pytest_addoption(parser):
    parser.addoption(
        '--catalog-url',
        default=DEFAULT_CATALOG_URL,
        help='URL for the intake catalog',
    )
    parser.addoption(
        '--catalog-key',
        default='UK',
        help="Sub-catalog key to select (empty string for none)",
    )
    parser.addoption(
        '--plot-dir',
        default=None,
        help=f'Directory to save plot outputs (default: {DEFAULT_PLOT_DIR})',
    )


@pytest.fixture(scope='session')
def plot_dir(request):
    opt = request.config.getoption('--plot-dir')
    path = Path(opt) if opt else DEFAULT_PLOT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope='session')
def get_pr_zooms(catalog):
    """Return a callable that fetches and caches pr at multiple zooms over a short time window."""
    cache = {}

    def _get(sim, freq, zooms=range(6)):
        key = (sim, freq)
        if key in cache:
            return cache[key]
        result = {}
        for zoom in zooms:
            try:
                ds = catalog[sim](zoom=zoom, time=freq).to_dask().pipe(hp_mods)
                result[zoom] = ds['pr'].isel(time=slice(0, 6)).compute()
            except Exception:
                pass
        cache[key] = result
        return result

    return _get


@pytest.fixture(scope='session')
def get_zoom0(catalog):
    """Return a callable that fetches and caches the full time series at zoom=0."""
    cache = {}

    def _get(sim, freq):
        key = (sim, freq)
        if key in cache:
            return cache[key]
        try:
            cache[key] = catalog[sim](zoom=0, time=freq).to_dask().pipe(hp_mods).compute()
        except Exception as exc:
            cache[key] = exc
        return cache[key]

    return _get


@pytest.fixture(scope='session')
def catalog(request):
    url = request.config.getoption('--catalog-url')
    key = request.config.getoption('--catalog-key')
    cat = intake.open_catalog(url)
    return cat[key] if key else cat


@pytest.fixture(scope='session')
def get_snapshot(catalog):
    """Return a callable that fetches and caches a single-timestep snapshot.

    Returns the computed xr.Dataset on success, or the Exception on failure.
    High-zoom datasets are sampled to HIGH_ZOOM_CELL_SAMPLE cells to keep
    runtime short.
    """
    cache = {}

    def _get(sim, zoom, freq):
        key = (sim, zoom, freq)
        if key in cache:
            return cache[key]

        try:
            ds = catalog[sim](zoom=zoom, time=freq).to_dask().pipe(hp_mods)
            snapshot = ds.isel(time=1)
            cache[key] = snapshot.compute()
        except Exception as exc:
            cache[key] = exc

        return cache[key]

    return _get
