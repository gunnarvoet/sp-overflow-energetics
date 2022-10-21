#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Plot Vertical Wave Flux Integrated
# for towyos and model

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pickle
import gvpy as gv
import nslib as nsl
# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'
# %autosave 300

# %%
cfg = nsl.io.load_config()

# %%
# Read towyo data, including energetics calcs
a = nsl.io.load_towyos()

# %% [markdown]
# Plot vertical internal wave fluxes.

# %%
print('plotting vertical internal wave fluxes')
nsl.plt.PlotVerticalEnergyFlux(cfg, a)
nsl.io.save_png('vertical_wave_flux_integrated')
nsl.io.save_pdf('vertical_wave_flux_integrated')

# %% [markdown]
# We may want to include the total vertical pressure work term in this plot as well. How does it look like when averaged in density classes?

# %%
vwfi = xr.open_dataarray(cfg.model.output.small_scale_vwf_integrated)
isotherms = vwfi.isot

E = xr.open_dataset(cfg.model.output.energy).swap_dims(Y='dist')

b = xr.open_dataset(cfg.model.output.data).swap_dims(Y='dist')

lateral_mask = (b.dist>0) & (b.dist<17)

# %% [markdown]
# For all time steps

# %%
vertPressureWork = E.IWEnergyFluxVert_sorted_rho.where(lateral_mask, drop=True).drop_vars(['T'])

th = b.th.where(lateral_mask, drop=True).drop_vars(['T'])

dyF = b.dyF.where(lateral_mask, drop=True)

tmp = xr.Dataset(dict(wppp=vertPressureWork))

tmp.coords['dyF'] = dyF

tmp['th'] = th


# %% [markdown]
# Let's integrate along the isotherms.

# %%
def int_along_isotherm(tmp, isot):
    interface = tmp.th.where(tmp.th < isot).argmax(dim="Z")
    return (tmp.wppp.isel(Z=interface) * tmp.dyF).sum(dim="dist")


# %%
def int_along_isotherms(tmp, isotherms):
    wppp_int = np.array([int_along_isotherm(tmp, isot) for isot in isotherms])
    return wppp_int


# %%
wppp_int = int_along_isotherms(tmp, isotherms)
wppp_int_da = xr.DataArray(wppp_int, coords=dict(isot=isotherms.data, T=b.time.data))
wppp_int_mean = wppp_int_da.mean(dim='T')

# %%
print('plotting vertical internal wave fluxes')
nsl.plt.PlotVerticalEnergyFluxWithPressureWork(cfg, wppp_int_mean, a)
nsl.io.save_png('vertical_wave_flux_and_vert_pressure_work_integrated')

# %%
nsl.plt.PlotVerticalEnergyFluxWithPressureWork(cfg, wppp_int_mean, a, for_pdf=True)
nsl.io.save_pdf('vertical_wave_flux_and_vert_pressure_work_integrated')

# %% [markdown]
# ### Temperature-density relationship

# %% [markdown]
# Plot relationship between towyo density and temperature to confirm the relationship between the axes above.

# %%
ty = a['t12']

# %%
fig, ax = gv.plot.quickfig()
_ = ax.scatter(ty.gsw_sigma4, ty.th, 1)
