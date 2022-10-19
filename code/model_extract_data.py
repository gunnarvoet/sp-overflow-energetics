import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import gsw

import gvpy as gv
import nslib as nsl

cfg = nsl.io.load_config()

# Read model data
print("reading model data")
b = nsl.model.load()

# Set identifier for this dataset. This is used to tag the APE data when saving/loading.
identifier = f"model{cfg.parameters.model.time_min}{cfg.parameters.model.time_max}"
b.attrs["modelrun"] = identifier
# Adjust distance to match with observations.
b["dist"] += 4.3

# Save reference density from model initialization
print("save reference density")
trange = range(5)
tmp = b.isel(T=trange)
tmp = nsl.model.calculate_density(tmp)
ref_rho = tmp.rho.isel(T=0, Y=-1)
ref_rho.to_netcdf(cfg.model.output.refrho)

print("subsample")
# Subsample
ti = np.flatnonzero((b.time > cfg.parameters.model.time_min) & (b.time < cfg.parameters.model.time_max))
b = b.isel(T=ti)

print("calculate density & pressure")
b = nsl.model.calculate_density(b)
b = nsl.model.calculate_pressure(b)
# if the following fails you may need to run model_sort_initial_density.ipynb
b = nsl.model.calculate_pressure_anomaly_sorted_rho(b, cfg)

# # Constants
# gravity = 9.81
# rho0 = 9.998000000000000e02  # from STDOUT, e.g.: grep -A 1 'rho' STDOUT.0000
# nuh = 1e-4
# kappah = 1e-4
# nuv = 1e-5
# kappav = 1e-5
# Cd = 1e-3

print("calculate velocities")
b = nsl.model.calculate_velocities(b)

# Save to netcdf
nsl.io.close_nc(cfg.model.output.data)
b.to_netcdf(cfg.model.output.data)
