import gsw
import gvpy as gv
import numpy as np
import scipy.signal as sg
import xarray as xr
from scipy.interpolate import interp1d

import nslib as nsl


def downstream_ctd_profile():
    """Generate a far-field density profile from CTD data for towyo APE
    calculation and save to project data directory."""
    cfg = nsl.io.load_config()
    ctd = nsl.io.load_ctd_profiles()

    # Define downstream profiles
    ds = ctd.where(
        (ctd.lat > -7.9) & (ctd.lon > -169.5) & (ctd.lon < -168.5), drop=True
    )
    mlat_downstream = ds.lat.mean().data
    sg_downstream = ds.sg4.mean(dim="x")
    ztmp = -gsw.z_from_p(ctd.p.mean(dim="x"), mlat_downstream)
    intf = interp1d(ztmp, sg_downstream, bounds_error=False)
    z = np.arange(3000, 6001, 1)
    sg_downstream = intf(z)

    # Low pass filter profile
    FilterOrder = 3  # Filter order
    Wn = 1 / 200.0  # Cutoff frequency (this is 100m if sampled at 1m)
    B, A = sg.butter(FilterOrder, Wn, output="ba")
    sg4filt = np.ones_like(sg_downstream) * np.nan
    sg4filt[~np.isnan(sg_downstream)] = sg.filtfilt(
        B, A, sg_downstream[~np.isnan(sg_downstream)]
    )

    # Generate xarray.Dataset and save as netcdf
    ctd_ds = xr.Dataset(
        data_vars=dict(sigma_4=(("z"), sg_downstream), sg4filt=(("z"), sg4filt)),
        coords=dict(z=z),
    )
    ctd_ds.attrs["title"] = "downstream_density_ctd_profile"
    print("saving downstream CTD profile...")
    nsl.io.close_nc(cfg.obs.output.downstream_density_nc)
    ctd_ds.to_netcdf(cfg.obs.output.downstream_density_nc)


if __name__ == "__main__":
    downstream_ctd_profile()
