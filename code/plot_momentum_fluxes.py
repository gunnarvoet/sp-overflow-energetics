#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Plot Momentum Fluxes
# for towyos and model

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# %% [markdown]
# Load towyo data

# %%
# Read towyo data, including energetics calcs
a = nsl.io.load_towyos()

# %% [markdown]
# Read model data

# %%
b = xr.open_dataset(cfg.model.output.data)
print('start: ', b.time[0].data, '\nend  : ', b.time[-1].data)

# %% [markdown]
# Calculate model momentum flux

# %%
momflux = nsl.model.rho0 * b.v * b.wp
momflux = momflux.swap_dims({'Y':'dist', 'T':'time'})

# %% [markdown]
# Select a subset for plotting and averaging

# %%
mflux = momflux.where((momflux.dist>0) & (momflux.dist<17) & (momflux.Z<-4000), drop=True)

# %% [markdown]
# Mean momentum flux (spatially averaged over km 0 to 17, and also in time).

# %%
fig, ax = gv.plot.quickfig(w=5)
mflux.mean(dim='dist').plot(y='Z', hue='time', add_legend=False, color='k', alpha=0.08);
mflux.mean(dim=['dist', 'time']).plot(y='Z', add_legend=False, color='w');
ax.set(xlabel='$F_M$ [N/m$^2$]');

# %% [markdown]
# The vertical momentum flux divergence is about 1 N/m$^2$ over 400$\,$m in the model.

# %% [markdown]
# Here are the towyo momentum fluxes. Slightly smaller than the model fluxes and more noisy but showing the same sign and diminish at the interface.

# %%
for n, ty in a.items():
    ty.vp_wp_mean_sg4class.plot(y='sg4bins', yincrease=False)

# %%
for n, ty in a.items():
    ty.vp_wp_mean_tclass.plot(y='tclass', yincrease=True)


# %% [markdown]
# Note that towyo momentum fluxes were calculated using baroclinic velocities based on local means. Let's also calculate these with the full baroclinic velocities.

# %%
def calc_momflux(ty):
    mf = 1045 * ty.u * ty.w
    ty['mf2'] = mf
    return ty
for k, ty in a.items():
    ty = calc_momflux(ty)


# %%
for k, ty in a.items():
    ty = ty.ty.IntegrateAlongIsopycnal('mf2')
# %%
fig, ax = gv.plot.quickfig(w=4)
for n, ty in a.items():
    ty.mf2_along_isopycnal.plot(y='sg4bins', yincrease=False)
# ax.set(ylim=(45.98, 45.92), xlim=(-0.8, 0.3))

# %% [markdown]
# Integrate model momentum flux along isotherms. Use algorithms from internal wave flux calculations.

# %%
subset = nsl.model_wave_fluxes.extract_subregion(b)

# %%
momflux = nsl.model.rho0 * subset.v * subset.wp
momflux.attrs["long_name"] = "momentum flux"
momflux.attrs["units"] = r"N/m$^2$"

# %%
momflux.isel(time=100).plot()

# %%
momfi = nsl.model_wave_fluxes.integrate_vertical_iw_flux(momflux, subset)

# %% [markdown]
# Plot horizontally integrated vertical momentum flux in temperature space vs time. Units are N/m.

# %%
momfi.plot()

# %%
momfi.mean(dim='time').plot(y='isot');

# %%
momfi.std(dim='time').plot(y='isot');

# %% [markdown]
# Note that model form drag was about $-1.6\times 10^4$ N/m.

# %% [markdown]
# Scale back from integrated momentum flux to average momentum flux by dividing by integration distance and plot the time-mean. Units here are N/m$^2$.

# %%
(momfi/17e3).mean(dim='time').plot(y='isot');

# %%
(momfi/17e3).std(dim='time').plot(y='isot');

# %% [markdown]
# Here the vertical divergence is about 2 N/m$^2$, probably also over something like 500m.

# %%
print(momfi.mean(dim='time').min().data)
print(momfi.min(dim='isot').std(dim='time').data)
print(momfi.mean(dim='time').min().data / 17e3)

# %%
_ = nsl.io.Res(
    name="ModelMomentumFlux",
    value=f"{momfi.mean(dim='time').min().data/1e4:1.1f}",
    unit="$10^4$\,N\,m$^{-1}$",
    comment="model absolute maximum momentum flux",
)

# %%
_ = nsl.io.Res(
    name="ModelMomentumFluxSigma",
    value=f"{momfi.min(dim='isot').std(dim='time').data/1e4:1.1f}",
    unit="$10^4$\,N\,m$^{-1}$",
    comment="model absolute maximum momentum flux standard deviation",
)

# %%
_ = nsl.io.Res(
    name="ModelMomentumFluxStress",
    value=f"{momfi.mean(dim='time').min().data/17e3:1.1f}",
    unit="N\,m$^{-2}$",
    comment="model absolute maximum momentum flux stress",
)

# %%
_ = nsl.io.Res(
    name="ModelMomentumFluxStressSigma",
    value=f"{momfi.min(dim='isot').std(dim='time').data/17e3:1.1f}",
    unit="N\,m$^{-2}$",
    comment="model absolute maximum momentum flux stress standard deviation",
)

# %%
for k, t in a.items():
    res_id = "TyA" if t.attrs["name"] == "2012" else "TyB"
    momflux = t.mf2_along_isopycnal.where(t.sg4bins>45.94).min().data
    print(momflux)
    _ = nsl.io.Res(
        name=res_id + "MomentumFlux",
        value=f"{momflux/1e4:1.1f}",
        unit="$10^4$\,N\,m$^{-1}$",
        comment="absolute maximum momentum flux",
    )
    average_momflux = t.mf2_along_isopycnal.where(t.sg4bins>45.94).min().data / 17e3
    print(average_momflux)
    _ = nsl.io.Res(
        name=res_id + "MomentumFluxAverage",
        value=f"{average_momflux:1.1f}",
        unit="N\,m$^{-2}$",
        comment="absolute maximum momentum flux",
    )

# %% [markdown]
# Plot both model and towyo momentum fluxes

# %%
nsl.plt.PlotMomentumFluxes(a, momfi)
nsl.io.save_png('momentum_flux')

# %%
nsl.plt.PlotMomentumFluxes(a, momfi, for_pdf=True)
nsl.io.save_pdf('momentum_flux')
