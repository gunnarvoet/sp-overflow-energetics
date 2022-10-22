#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# #### Imports

# %%
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import gsw
import os
import xarray as xr
import pandas as pd
import pickle
import cmocean

import gvpy as gv
import nslib as nsl

# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'
# %autosave 300
# %%
gv.misc.warnless()

# %% [markdown]
# Load some extra colors.
# %%
col, col2 = nsl.plt.load_colors()
# %% [markdown]
# Load configuration from `config.yml`.

# %%
cfg = nsl.io.load_config()

# %% [markdown]
# # Towyo Energy Calculations
# %% [markdown]
# Load towyo data in xarray format. Towyo data were read from .mat file and converted to an xarray Dataset in [01_northern_sill_load_and_format_towyo_data](./01_northern_sill_load_and_format_towyo_data.ipynb). Comes as sorted dict ``a`` with two entries, one is the 2012 and the other one is the 2014 towyo.
# %%
print('reading towyo data')
a = nsl.io.load_towyos(base=True)
# %% [markdown]
# ## Gradients
# %% [markdown]
# Horizontal temperature / isopycnal gradients show lee wave-like structure. We'll also need the horizontal gradients to derive vertical velocity from $\eta$.
# %%
print('calculate horizontal gradients')
[t.ty.HorizontalGradients() for k, t in a.items()];
# %% [markdown]
# ## Volume flux
# %% [markdown]
# Integrate northward velocity below interface.
# %%
print('calculate volume flux')
[t.ty.VerticalIntegralLowerLayer('v', 45.94) for k, t in a.items()];
# %% [markdown]
# ## Energetics

# %% [markdown]
# ### Kinetic Energy
# %%
print('calculate kinetic energy')
[t.ty.KineticEnergy() for k, t in a.items()];
# %% [markdown]
# ### Available Potential Energy
# Following [Lamb(2008)](/Users/gunnar/Projects/wavebib/articles/Lamb/Lamb2008.pdf) one can use the available potential energy density $$ E_a(x,z,t) = g \int_z^{z^{\ast}(x,z,t)} (\overline{\rho}(s) - \rho(x,z,t)) ds $$ to calculate APE. Here, $\overline{\rho}$ is a far-field density profile and $z^{\ast}(x,z,t)$ is the height of the fluid parcel at (x,z) at time $t$ in the reference state. See also [Winters et al. (1995)](~/Projects/wavebib/articles/Winters/Winters1995.pdf) for APE.
# %%
print('calculate APE')
# Load downstream (far field) density profile generated in `01_downstream_density.py`.
ctd = xr.open_dataset(cfg.obs.output.downstream_density_nc)
# Let's make sure we don't have to re-run this every time since it is slow.
# Check if we can load pre-existing data, otherwise run and save.
try:
    ape12 = xr.open_dataarray(cfg.obs.output.ape12_nc)
    ape14 = xr.open_dataarray(cfg.obs.output.ape14_nc)
    a['t12']['APE'] = ape12
    a['t14']['APE'] = ape14
    # update the time stamp on the files, otherwise make will think it has to run again
    cfg.obs.output.ape12_nc.touch()
    cfg.obs.output.ape14_nc.touch()
except:
    [t.ty.AvailablePotentialEnergy(ctd) for k, t in a.items()]
    ape12 = a['t12'].APE
    ape14 = a['t14'].APE
    ape12.to_netcdf(cfg.obs.output.ape12_nc)
    ape14.to_netcdf(cfg.obs.output.ape14_nc)
# %% [markdown]
# Calculate a second version of APE based on the resorted density field from the model initialization. Since the model has different density coordinates, we find an offset factor to relate $\sigma_4$ in the towyo data to $\rho$ in the sorted model data.

# %%
rhos = xr.open_dataarray(cfg.model.output.refrho_sorted)
rhos['Z'] = -1 * rhos.Z
# offset determined via visual matching of profiles above the overflow - see plot below
scale_factor = 953.688
rhos = rhos - scale_factor
# Interpolate to 1m vertical resolution
znew = a['t12'].z.copy()
rhos = rhos.interp(Z=znew)
rhos.attrs["title"] = "refrho_sorted"

# %%
fig, ax = gv.plot.quickfig(w=5)
rhos.plot(color='C1', y='Z')
(a['t12'].gsw_sigma4.isel(x=0)).plot(color='C3', y='z')
(a['t12'].gsw_sigma4.isel(x=-1)).plot(color='C3', y='z')
(a['t14'].gsw_sigma4.isel(x=0)).plot(color='C0', y='z')
(a['t14'].gsw_sigma4.isel(x=-1)).plot(color='C0', y='z')
ax.plot(ctd['sg4filt'], ctd['z'], color='0.5')
ax.set(xlim=(45.87, 45.99), ylim=(5500, 3500), title='');

# %%
print('calculate APE against sorted model density')
try:
    ape12s = xr.open_dataarray(cfg.obs.output.ape12s_nc)
    ape14s = xr.open_dataarray(cfg.obs.output.ape14s_nc)
    a['t12']['APEs'] = ape12s
    a['t14']['APEs'] = ape14s
    # update the time stamp on the files, otherwise make will think it has to run again
    cfg.obs.output.ape12s_nc.touch()
    cfg.obs.output.ape14s_nc.touch()
except:
    [t.ty.AvailablePotentialEnergy(rhos) for k, t in a.items()]
    ape12s = a['t12'].APEs
    ape14s = a['t14'].APEs
    ape12s.to_netcdf(cfg.obs.output.ape12s_nc)
    ape14s.to_netcdf(cfg.obs.output.ape14s_nc)
# %% [markdown]
# ### KE, PE fluxes
# %%
print('calculate energy fluxes')
[t.ty.KE_flux() for k, t in a.items()]
[t.ty.APE_flux() for k, t in a.items()];
# %% [markdown]
# ### Internal wave fluxes
# %%
print('calculate internal wave fluxes')
# Calculate $w' p'$ to estimate the vertical energy flux of the lee waves.
[t.ty.MeanDensityProfile() for k, t in a.items()];
# %% [markdown]
# ### Windowed mean density
# Average within +/- 2.5km. Low-pass filter as well.
# %%
[t.ty.WindowedMeanDensityProfile(2.5) for k, t in a.items()];
# %% [markdown]
# ### Density anomaly
# Calculate the density anomaly $\rho'$. It turns out to be better to use a windowed mean density profile instead of an overall mean density profile.
# %%
[t.ty.DensityAnomaly() for k, t in a.items()];
# %% [markdown]
# Also calculate a density anomaly based on the reference profile from sorted intial model density.

# %%
[t.ty.DensityAnomalySortedReference(rhos) for k, t in a.items()];
# %% [markdown]
# ### Pressure anomaly
# We need to start integrating at a depth level where we have data in each profile, otherwise this will get messy. Only using the windowed mean density here as the other one is too skewed. Units: 1 N/m$^2$ = 10$^{-4}$dbar
#
# find start depth (first row without nan's) in 2012 towyo, then use this for both as 2014 was shallower than 2012.
# %%
ty = a["t12"]
xx = np.where(ty.dist.values < 25)
StartIndex = 0
while np.sum(np.where(np.isnan(ty.rho_anom_windowed.values[StartIndex, xx]))):
    StartIndex = StartIndex + 1
# print('start index {}\nstart depth {}m'.format(StartIndex, ty.z.values[StartIndex]))

r = nsl.io.Res(
    "ty_int_start_depth",
    ty.z.values[StartIndex],
    unit="m",
    comment="upper integration limit for pressure anomaly",
)

[t.ty.PressureAnomaly(StartIndex) for k, t in a.items()]
# %%
ty = a['t12']

# %% [markdown]
# Compare the two pressure anomalies (sorted vs true downstream density reference).

# %%
ty.p_anom.dropna(dim='z', how='all').plot(yincrease=False)

# %%
ty.p_anom_s.dropna(dim='z', how='all').plot(yincrease=False)

# %%
(ty.p_anom - ty.p_anom_s).dropna(dim='z', how='all').plot(yincrease=False)

# %% [markdown]
# ### Isopycnal displacements
# Calculate ispycnal displacement $\eta$ - find distance of each value to isopycnal depth in mean density profile.
#
# TODO: Can we do better in the bottom layer here? Data drop out faster than they should. Seems like the highest densities are not represented in the mean density...
# %%
[t.ty.IsopycnalDisplacement() for k, t in a.items()];
# %% [markdown]
# ### Vertical velocity
# Calculate from $\eta$: $w=d\eta/dt=u\ d\eta/dx$ Using both constant $v$ and observed $v$ below.
#
# TODO: Smoothing in the vertical - low-pass filter over 10m and then do the diff over 10m
# %%
[t.ty.VerticalVelocityFromEta() for k, t in a.items()];
# %% [markdown]
# ### Mean horizontal and vertical velocity
# Smoothed with a low pass filter
# %%
[t.ty.MeanVelocities() for k, t in a.items()];
# %% [markdown]
# ### Windowed mean velocities
# Using same window size (+/- 2.5km) as above.
# %%
[t.ty.WindowedMeanVelocities(deltax=2.5) for k, t in a.items()];
# %% [markdown]
# ### Velocity anomalies
# Using the windowed mean velocity fields here.
#
# TODO: Explore this in isopycnal coordinates?
# %%
[t.ty.VelocityAnomalies() for k, t in a.items()];
# %% [markdown]
# ### Internal wave energy fluxes
#
# Small-scale internal wave fluxes $w'' p''$ and $v'' p''$ from the window-based primed quantities.
#
# Pressure work terms calculated from $p'$ based on the sorted model reference density and $v'$ as the full baroclinic velocity.
# %%
[t.ty.WaveEnergyFluxes() for k, t in a.items()];
# %% [markdown]
# ### Average energy fluxes in temperature and density classes
# %%
print('average energy fluxes in temperature and density space')
def CalculateAverageEnergyFluxTClass(ty,tt,deltat):
    if 'tclass' in ty:
        ty = ty.drop('tclass')
    wp_pp_mean_tclass = []
    vp_pp_mean_tclass = []
    tmp = ty.where(ty.dist<25)
    for j, t in enumerate(tt):
        mw = np.nanmean(tmp.wp_pp.values[(tmp.PT.values>=t) & (tmp.PT.values<t+deltat)])
        wp_pp_mean_tclass.append(mw)
        mv = np.nanmean(tmp.vp_pp.values[(tmp.PT.values>=t) & (tmp.PT.values<t+deltat)])
        vp_pp_mean_tclass.append(mv)
    ty['wp_pp_mean_tclass'] = (['tclass'], wp_pp_mean_tclass)
    ty['vp_pp_mean_tclass'] = (['tclass'], vp_pp_mean_tclass)
    ty.coords['tclass'] = (['tclass'], tt)
    return ty
def CalculateAverageEnergyFluxSGClass(ty,sgbins,deltasg):
    if 'sg4bins' in ty:
        ty = ty.drop('sg4bins')
    wp_pp_mean_sg4class = []
    vp_pp_mean_sg4class = []
    tmp = ty.where(ty.dist<25)
    for j, sg in enumerate(sgbins):
        mw = np.nanmean(tmp.wp_pp.values[(tmp.gsw_sigma4.values>=sg) & (tmp.gsw_sigma4.values<sg+deltasg)])
        wp_pp_mean_sg4class.append(mw)
        mv = np.nanmean(tmp.vp_pp.values[(tmp.gsw_sigma4.values>=sg) & (tmp.gsw_sigma4.values<sg+deltasg)])
        vp_pp_mean_sg4class.append(mv)
    ty['wp_pp_mean_sg4class'] = (['sg4bins'], wp_pp_mean_sg4class)
    ty['vp_pp_mean_sg4class'] = (['sg4bins'], vp_pp_mean_sg4class)
    ty.coords['sg4bins'] = (['sg4bins'], sgbins)
    return ty
deltasg4 = 0.01
sg4_bins = np.arange(45.89, 45.98+deltasg4, deltasg4)
deltat = 0.05
tt = np.arange(0.65,1.2+deltat,deltat)
# apply to both towyos
for k, ty in a.items():
    ty = CalculateAverageEnergyFluxTClass(ty,tt,deltat)
for k, ty in a.items():
    ty = CalculateAverageEnergyFluxSGClass(ty,sg4_bins,deltasg4)
# %% [markdown]
# ### Vertical momentum flux
# %%
print('calculate vertical momentum flux')
# Momentum flux is $\rho_0 u'w'$ [N/m$^2$].
[t.ty.MomentumFluxes() for k, t in a.items()];
# %%
# Vertical momentum flux in temperature and density classes
def CalculateAverageVerticalMomentumFluxTClass(ty,tt,deltat):
    vp_wp_mean_tclass = []
    tmp = ty.where(ty.dist<25)
    for j, t in enumerate(tt):
        mw = np.nanmean(tmp.vp_wp.values[(tmp.PT.values>=t) & (tmp.PT.values<t+deltat)])
        vp_wp_mean_tclass.append(mw)
    ty['vp_wp_mean_tclass'] = (['tclass'], vp_wp_mean_tclass)
    return ty
def CalculateAverageVerticalMomentumFluxSGClass(ty,sg4_bins,deltasg4):
    vp_wp_mean_sg4class = []
    tmp = ty.where(ty.dist<25)
    for j, t in enumerate(sg4_bins):
        mw = np.nanmean(tmp.vp_wp.values[(tmp.gsw_sigma4.values>=t) & (tmp.gsw_sigma4.values<t+deltasg4)])
        vp_wp_mean_sg4class.append(mw)
    ty['vp_wp_mean_sg4class'] = (['sg4bins'], vp_wp_mean_sg4class)
    return ty

# apply to both towyos
deltat = 0.05
for k, ty in a.items():
    tt = ty.tclass.values
    ty = CalculateAverageVerticalMomentumFluxTClass(ty,tt,deltat)
    sg4_bins = ty.sg4bins.values
    ty = CalculateAverageVerticalMomentumFluxSGClass(ty,sg4_bins,deltasg4)
# %% [markdown]
# ## Integrals

# %% [markdown]
# ### Turbulent dissipation
#
# We need to fill in background values where no overturns detected, bottom must stay nan.
# %%
print('fill background dissipation')
[t.ty.FillInBackgroundDissipation() for k, t in a.items()];

# %% [markdown]
# Now integrate dissipation in the vertical.

# %%
print('integrate dissipation')
[t.ty.IntegrateDissipation(cfg) for k, t in a.items()];

# %%
InterfaceSG4 = cfg.parameters.towyo.interfacesg4
print(f'upper interface in towyo energy calcs is {InterfaceSG4} (sigma4)')
# nsl.io.res_save(ty_upper_int_limit=InterfaceSG4)
r = nsl.io.Res(
    name="TyUpperIntLimit",
    value=InterfaceSG4,
    unit='kg\,m$^{-3}$',
    comment="upper integration limit (sigma4) for budgets",
)

# %% [markdown]
# ### Energy
# %%
print('integrate energy in layers')
[t.ty.VerticalIntegralLowerLayer('KE', InterfaceSG4) for k, t in a.items()]
[t.ty.VerticalIntegralLowerLayer('APE', InterfaceSG4) for k, t in a.items()]
[t.ty.VerticalIntegralLowerLayer('APEs', InterfaceSG4) for k, t in a.items()];


# %% [markdown]
# ### Energy fluxes
# %%
print('integrate energy fluxes in layers')
[t.ty.VerticalIntegralLowerLayer('KEflux', InterfaceSG4) for k, t in a.items()]
[t.ty.VerticalIntegralLowerLayer('APEflux', InterfaceSG4) for k, t in a.items()]
[t.ty.VerticalIntegralLowerLayer('APEsflux', InterfaceSG4) for k, t in a.items()];


# %%
ty = a['t12'].swap_dims({'x':'dist'})
ty.APEsfluxVI.plot()

# %% [markdown]
# ### Horizontal pressure work
# %%
print('integrate horizontal pressure work over layer')
[t.ty.VerticalIntegralLowerLayer('PressureWorkHoriz', InterfaceSG4) for k, t in a.items()];
[t.ty.VerticalIntegralLowerLayer('PressureWorkHorizSorted', InterfaceSG4) for k, t in a.items()];

# %%
ty = a['t12'].swap_dims({'x':'dist'})
ty.PressureWorkHorizVI.plot();
ty.PressureWorkHorizSortedVI.plot();

# %%
ty = a['t14'].swap_dims({'x':'dist'})
ty.PressureWorkHorizVI.plot();
ty.PressureWorkHorizSortedVI.plot();

# %% [markdown]
# ### Vertical Internal Wave Fluxes
# %%
ty = a['t14']
ty.PressureWorkVert.plot()

# %%
print('integrate IW fluxes along isopycnals')
# Integrate the upward wave energy flux along isopycnals.
# Average in an isopycnal band and then integrate.
[t.ty.IntegrateAlongIsopycnal('wp_pp', distmax=17) for k, t in a.items()];
[t.ty.IntegrateAlongIsopycnal('PressureWorkVert', distmax=17) for k, t in a.items()];
[t.ty.IntegrateAlongIsopycnal('PressureWorkVertSorted', distmax=17) for k, t in a.items()];


# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 5),
                       constrained_layout=True, sharey=True)
for axi, (n, ty) in zip(ax, a.items()):
    ty.PressureWorkVert_along_isopycnal.plot(ax=axi, y='sg4bins')
    ty.PressureWorkVertSorted_along_isopycnal.plot(ax=axi, y='sg4bins')
    ty.wp_pp_along_isopycnal.plot(ax=axi, y='sg4bins', yincrease=False)

# %% [markdown]
# ## Vertical energy wave flux divergence

# %% [markdown]
# See if the vertical wave energy flux divergence goes into local dissipation of turbulent kinetic energy: $$\frac{\partial \left< w^\prime p^\prime \right> }{\partial z} = \rho \epsilon$$ We can do this either in depth or temperature/density space and could either sum/average over the whole section or look at smaller chunks.
#
# TODO: - how can we do this in temperature/density space? - do this in smaller horizontal chunks!
#
# If we integrate epsilon in density layers as well we can compare it better to the vertical divergence of the internal wave energy flux in density classes.

# %%
ty = a['t14']
wppp_grad = np.gradient(ty.wp_pp, ty.z, axis=0)
ty['wp_pp_grad'] = (['z', 'x'], wppp_grad)
ty.ty.AverageAlongIsopycnal('wp_pp_grad')
ty.ty.AverageAlongIsopycnal('eps')


# %% [markdown]
# ## Form Drag

# %% [markdown]
# Calculate form drag between kilometer 0 and kilometer 17.

# %%
print('=== FORM DRAG ======================')
[t.ty.FormDrag(maxdist=17) for k, t in a.items()];
print('====================================')

# %% [markdown]
# ## Bottom Drag

# %% [markdown]
# Drag coefficient $C_D = (1 - 3)\times10^{-3}$.
#
# Bottom stress $\tau_B = \rho C_D u^2$ [N/m$^2$].
#
# Velocity $v$ is the free flowing velocity just outside the stress layer.
#
# Energy lost to bottom stress $\tau_B u$ [W/m$^2$].
#
# Integrating along the flow results in units of [W/m].

# %%
[t.ty.BottomDrag(cd=2e-3) for k, t in a.items()];

# %% [markdown]
# ## Energy Budget Terms

# %%
budget_terms = [nsl.towyo.calculate_energy_budget_terms(ty, cfg) for k, ty in a.items()]

# %% [markdown]
# Combine the results into one dataset.

# %%
bt = xr.concat(budget_terms, dim='year')
bt.coords['year'] = [2012, 2014]

# %% [markdown]
# Calculate the residual. We don't want to include the vertical wave flux term ($w'' p''$) as this would be double-counting and should already be included in the vertical pressure work term.

# %%
residual = bt.apesflux_div + bt.keflux_div + bt.dissipation + bt.bottom_dissipation + bt.vertical_pressure_work_sorted + bt.horizontal_pressure_work_sorted

# %%
residual

# %%
_ = nsl.io.Res(
    name="TyAResidual",
    value=f"{-residual.sel(year=2012).data/1e3:1.1f}",
    unit="kW\,m$^{-1}$",
    comment="towyo budget residual",
)
_ = nsl.io.Res(
    name="TyBResidual",
    value=f"{-residual.sel(year=2014).data/1e3:1.1f}",
    unit="kW\,m$^{-1}$",
    comment="towyo budget residual",
)

# %% [markdown]
# ## Form Drag Energy Loss

# %% [markdown]
# What velocity would be needed to match form drag energy loss with the observed energy loss terms in the energy budget?

# %% [markdown]
# $$D_f * v = \epsilon$$
# $$v = \epsilon / D_f$$

# %% [markdown]
# The loss terms we want to consider here are turbulent dissipation, dissipation due to bottom friction, and vertical pressure work divergence. In other flow situations we would also want to include the horizontal pressure work divergence, but since it does work on the flow and is an energy source in our case that doesn't make much sense.

# %%
[nsl.towyo.calculate_matching_form_drag_velocity(a, bt, year) for year in [2012, 2014]]

# %% [markdown]
# Upstream velocities need to be 10-15cm/s.

# %% [markdown]
# ## Save

# %%
nsl.io.close_nc(cfg.obs.output.energy_budget)
bt.to_netcdf(cfg.obs.output.energy_budget)

# %% [markdown]
# Run another energy budget extended further downstream (km 20 to 25).

# %%
print("\n----------\nExtended budget to km 25\n----------")

# %%
config2 = nsl.io.load_config()

# %%
config2.parameters.towyo.energy_budget_dnstream_range

# %%
config2.parameters.towyo.energy_budget_dnstream_range = [20, 25]

# %%
config2.parameters.towyo.energy_budget_dnstream_range

# %%
budget_terms = [
    nsl.towyo.calculate_energy_budget_terms(ty, config2, save_results=False)
    for k, ty in a.items()
]

# %% [markdown]
# Combine the results into one dataset.

# %%
bt2 = xr.concat(budget_terms, dim='year')
bt2.coords['year'] = [2012, 2014]

# %% [markdown]
# Calculate the residual. We don't want to include the vertical wave flux term ($w'' p''$) as this would be double-counting and should already be included in the vertical pressure work term.

# %%
bt2.apesflux_div + bt2.keflux_div + bt2.dissipation + bt2.bottom_dissipation + bt2.vertical_pressure_work_sorted + bt2.horizontal_pressure_work_sorted

# %% [markdown]
# ## Save data and results


# %%
print('\n\n----------\nsave to netcdf\n----------')
ty = a['t12']
nsl.io.close_nc(cfg.obs.output.towyo_2012)
ty.to_netcdf(cfg.obs.output.towyo_2012)
ty = a['t14']
nsl.io.close_nc(cfg.obs.output.towyo_2014)
ty.to_netcdf(cfg.obs.output.towyo_2014)

