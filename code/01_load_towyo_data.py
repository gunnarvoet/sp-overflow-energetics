"""
Load and format towyo data for further analysis.
"""

import collections

import xarray as xr

import nslib as nsl


def read_towyos():
    t12 = nsl.io.load_towyo(2012, 104)

    t141 = nsl.io.load_towyo(2014, 11)
    t142 = nsl.io.load_towyo(2014, 12)
    t14 = xr.concat([t141, t142], dim="x")

    a = {}
    a["t12"] = t12
    a["t14"] = t14
    a["t12"].attrs["name"] = "2012"
    a["t14"].attrs["name"] = "2014"
    # Ordered dict so the towyos are ordered by name.
    a = collections.OrderedDict(sorted(a.items()))

    return a


def add_derived_and_aux_data(a):
    # Run a few methods to enhance the dataset. These methods were added by
    # importing nslib.
    [t.ty.AddTopo() for k, t in a.items()]
    [t.ty.AddBathyMask() for k, t in a.items()]
    [t.ty.AddEpsOverturns() for k, t in a.items()]
    [t.ty.AddDensity() for k, t in a.items()]
    [t.ty.CalculateSpeedDirection() for k, t in a.items()]
    [t.ty.BoundaryLayerMask() for k, t in a.items()]
    [t.ty.FillBottomValuesDensity() for k, t in a.items()]
    return a


def save_data(a):
    cfg = nsl.io.load_config()

    # Remove output netcdf files if they exist
    nsl.io.close_nc(cfg.obs.output.towyo_2012_in)
    nsl.io.close_nc(cfg.obs.output.towyo_2014_in)

    # Save data
    print("save to netcdf")
    a["t12"].to_netcdf(cfg.obs.output.towyo_2012_in)
    a["t14"].to_netcdf(cfg.obs.output.towyo_2014_in)


def run_all():
    a = read_towyos()
    a = add_derived_and_aux_data(a)
    save_data(a)


if __name__ == "__main__":
    run_all()
