import pytest
import matplotlib.pyplot as plt

from datasets import FREQS, HK26_SIMS, ZOOMS
import um_to_healpix.plotting as umplt

pytestmark = pytest.mark.plots


def _sim_short(sim):
    return sim.replace('um_', '').replace('_hk26', '')


PARAMS = [(sim, zoom, freq) for sim in HK26_SIMS for zoom in ZOOMS for freq in FREQS]
PARAM_IDS = [f"{_sim_short(sim)}-z{zoom}-{freq}" for sim, zoom, freq in PARAMS]

SIM_FREQ_PARAMS = [(sim, freq) for sim in HK26_SIMS for freq in FREQS]
SIM_FREQ_IDS = [f"{_sim_short(sim)}-{freq}" for sim, freq in SIM_FREQ_PARAMS]

SIM_ZOOM_PARAMS = [(sim, zoom) for sim in HK26_SIMS for zoom in ZOOMS]
SIM_ZOOM_IDS = [f"{_sim_short(sim)}-z{zoom}" for sim, zoom in SIM_ZOOM_PARAMS]


@pytest.mark.parametrize("sim,zoom,freq", PARAMS, ids=PARAM_IDS)
def test_plot_all_fields(get_snapshot, plot_dir, sim, zoom, freq):
    snapshot = get_snapshot(sim, zoom, freq)
    if isinstance(snapshot, Exception):
        pytest.skip(f"Not available: {snapshot}")

    out = plot_dir / _sim_short(sim) / freq
    out.mkdir(parents=True, exist_ok=True)

    if 'pressure' in snapshot.dims:
        snapshot = snapshot.sel(pressure=500, method='nearest')
    umplt.plot_all_fields(snapshot)
    path = out / f'all_fields.z{zoom}.png'
    plt.savefig(path)
    plt.close('all')

    assert path.exists()


@pytest.mark.parametrize("sim,freq", SIM_FREQ_PARAMS, ids=SIM_FREQ_IDS)
def test_plot_pr_zonal_mean(get_pr_zooms, plot_dir, sim, freq):
    pr_zooms = get_pr_zooms(sim, freq)
    if not pr_zooms:
        pytest.skip("No pr data available")

    out = plot_dir / _sim_short(sim) / freq
    out.mkdir(parents=True, exist_ok=True)

    umplt.plot_zonal_mean(pr_zooms.values())
    path = out / 'pr_zonal_mean.png'
    plt.savefig(path)
    plt.close('all')

    assert path.exists()


@pytest.mark.parametrize("sim,freq", SIM_FREQ_PARAMS, ids=SIM_FREQ_IDS)
def test_plot_pr_timeseries(get_pr_zooms, plot_dir, sim, freq):
    pr_zooms = get_pr_zooms(sim, freq)
    if not pr_zooms:
        pytest.skip("No pr data available")

    out = plot_dir / _sim_short(sim) / freq
    out.mkdir(parents=True, exist_ok=True)

    umplt.plot_timeseries(pr_zooms)
    path = out / 'pr_timeseries.png'
    plt.savefig(path)
    plt.close('all')

    assert path.exists()


@pytest.mark.parametrize("sim,zoom", SIM_ZOOM_PARAMS, ids=SIM_ZOOM_IDS)
def test_plot_clw_pressure_profile(get_snapshot, plot_dir, sim, zoom):
    snapshot = get_snapshot(sim, zoom, 'PT3H')
    if isinstance(snapshot, Exception):
        pytest.skip(f"Not available: {snapshot}")
    if 'clw' not in snapshot:
        pytest.skip("clw not in dataset")

    out = plot_dir / _sim_short(sim) / 'PT3H'
    out.mkdir(parents=True, exist_ok=True)

    clw = snapshot['clw']
    plt.figure()
    plt.plot(clw.mean(dim='cell'), clw.pressure)
    plt.ylim((1000, 0))
    plt.xlabel(f'{clw.long_name} ({clw.attrs.get("units", "-")})')
    plt.title(f'z{zoom}')
    path = out / f'clw_pressure_profile.z{zoom}.png'
    plt.savefig(path)
    plt.close('all')

    assert path.exists()
