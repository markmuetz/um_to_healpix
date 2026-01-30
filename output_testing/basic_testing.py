import xarray as xr
import easygems.healpix as egh
import matplotlib.pyplot as plt

from um_to_healpix.util import load_config

if __name__ == '__main__':
    config = load_config('../config/hk26_config.py')

    sim = 'glm.n2560_RAL3p3.tuned'
    freq = 'PT1H'  # 2D
    # freq = 'PT3H'  # 3D
    zoom = 10  # Only zoom with any data in so far.
    on_jasmin = False
    if on_jasmin:
        protocol = 'http'
        baseurl = 'hackathon-o.s3.jc.rl.ac.uk'
    else:
        protocol = 'https'
        baseurl = 'hackathon-o.s3-ext.jc.rl.ac.uk'
    url  = f'{protocol}://{baseurl}/sim-data/{config.deploy}/{config.output_vn}/{sim}/um.{freq}.hp_z{zoom}.zarr/'
    print(url)

    ds = xr.open_dataset(url, engine='zarr')

    egh.healpix_show(ds.tas.sel(time='2020-01-20 11:00'))
    plt.show()