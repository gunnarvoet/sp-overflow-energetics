# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# #### Imports

# %%
# %matplotlib inline
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import gsw

import gvpy as gv
import nslib as nsl

# %config InlineBackend.figure_format = 'retina'

# %reload_ext autoreload
# %autoreload 2
# %autosave 300

# %%
is_notebook = gv.misc.is_notebook()

# %% [markdown]
# # Model Form Drag
#
# Calculate form drag for the Northern Sill model run.

# %%
cfg = nsl.io.load_config()

# %% [markdown]
# ## Read model data

# %%
b = xr.open_dataset(cfg.model.output.data)

# %%
print('start: ', b.time[0].data, '\nend  : ', b.time[-1].data)

# %%
E = xr.open_dataset(cfg.model.output.energy)

# %% [markdown]
# some constants

# %%
# constants
gravity = 9.81
rho0 = 9.998000000000000e02  # from STDOUT, e.g.: grep -A 1 'rho' STDOUT.0000

"""
Horizontal eddy viscosity (`viscAh`)
$$\nu_H=10^{-4} \mathrm{m}^2/\mathrm{s}$$
Horizontal diffusivity of heat (and thereby mass with the LES, `diffKhT`):
$$\kappa_H=10^{-4} \mathrm{m}^2/\mathrm{s}$$
and vertical equivalents.
"""
nuh = 1e-4
kappah = 1e-4
nuv = 1e-5
kappav = 1e-5

# Bottom drag coefficient
Cd = 1e-3

# %% [markdown]
# ## Form Drag

# %% [markdown]
# $$D_f = -\int p_B^\prime \frac{dh}{dx} dx$$

# %% [markdown]
# Bottom pressure anomaly is zero at the southern end of the domain. Density and thus pressure are calculated relative to the reference density profile. Since we are integrating between equal depths it doesn't matter though whether there is any offset in p'.

# %%
TopoGradient = np.gradient(-1*b.Depth,b.Y)

# %% [markdown]
# Plot form drag parameters

# %%
if is_notebook:
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), sharex='col')
    ax = ax.flatten()
    for axi in ax:
        gv.plot.axstyle(axi)

    for axi in ax[0:2]:
        axi.plot(b.dist,(b.phibot).isel(T=0), label='start')
        axi.plot(b.dist,(b.phibot).isel(T=100), label='middle')
        axi.plot(b.dist,(b.phibot).isel(T=-1), label='end')
        axi.set(ylabel='p$^\prime$ [{}]'.format(b.phibot.attrs['units']),
                  title='Bottom Pressure Anomaly p$^\prime$')
        axi.legend();

    for axi in ax[2:4]:
        axi.plot(b.dist, TopoGradient)
        axi.set(ylabel='db/dx', title='Topographic Gradient');

    for axi in ax[4:6]:
        axi.plot(b.dist,b.Depth)
        axi.plot([0, 17], np.tile(5000, 2), 'ro')
    #     axi.plot([-3.5, 11.5], np.tile(4990, 2), 'ko')
        axi.set(ylim=(5300, 4600))
        axi.set(xlabel='Distance [km]', ylabel='Depth [m]', title='Bathymetry')

    for axi in ax[1::2]:
        axi.set(xlim=(-10, 45))

# %% [markdown]
# Product of bottom pressure anomaly and topography gradient.

# %%
FormDragTMP = -1*(b.phibot)*TopoGradient

# %% [markdown]
# Integrate from left to right at equal depths. Bathymetry has been lined up with the towyo, so we'll use the same integration limits.

# %%
mask = ((b.dist>=0) & (b.dist<17))
FormDrag = np.trapz(FormDragTMP.isel(Y=mask),b.Y[mask])
FormDragStress = FormDrag/17e3

# %%
print('Form Drag is {:1.1f} N/m'.format(FormDrag.mean()))
print('Stress associated with form drag is {:1.4f} N/m^2'.format(FormDragStress.mean()))

# %%
FormDrag.std()

# %%
FormDragStress.std()

# %% [markdown]
# Save to latex results file.

# %%
_ = nsl.io.Res(
    name="ModelFormDrag",
    value=f"{FormDrag.mean()/1e4:1.1f}",
    unit="$10^4$\,N\,m$^{-1}$",
    comment="model form drag",
)

# %%
_ = nsl.io.Res(
    name="ModelFormDragSigma",
    value=f"{FormDrag.std()/1e4:1.1f}",
    unit="$10^4$\,N\,m$^{-1}$",
    comment="model form drag standard deviation",
)

# %%
_ = nsl.io.Res(
    name="ModelFormDragStress",
    value=f"{FormDragStress.mean():1.1f}",
    unit="N\,m$^{-2}$",
    comment="model form drag stress",
)

# %%
_ = nsl.io.Res(
    name="ModelFormDragStressSigma",
    value=f"{FormDragStress.std():1.1f}",
    unit="N\,m$^{-2}$",
    comment="model form drag stress standard deviation",
)

# %%
if is_notebook:
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), sharex=True)
    for axi in ax:
        gv.plot.axstyle(axi)
    ax[0].plot(b.time,FormDrag)
    ax[0].set(ylabel='D$_F$ [N/m]')

    ax[1].plot(b.time,FormDragStress)
    ax[1].set(ylabel=r'$\tau_F$ [N/m$^2$]',xlabel='time [hrs]');

# %%
U0=0.1
print('Form drag power loss is {:1.1f} W/m'.format(FormDrag.mean()*U0))
print('Form drag average power loss is {:1.3f} W/m^2'.format(FormDrag.mean()*U0/17e3))

# %% [markdown]
# Loss terms are 2.2 and 1.3kW/m. Velocities have to be between 8 [0.9deg layer] and 13 cm/s [0.8deg layer] to match.

# %%
vel08h = -2200 / FormDrag.mean()
print(vel08h)

# %%
_ = nsl.io.Res(
    name="ModelLowerFormDragMatchingVelocity",
    value=f"{vel08h:1.2f}",
    unit="m\,s$^{-1}$",
    comment="model form drag matching velocity",
)

# %%
vel09h = -1300 / FormDrag.mean()
print(vel09h)

# %%
_ = nsl.io.Res(
    name="ModelUpperFormDragMatchingVelocity",
    value=f"{vel09h:1.2f}",
    unit="m\,s$^{-1}$",
    comment="model form drag matching velocity",
)

# %% [markdown]
# Let's have a look at upstream velocities by averaging over the dense layer. Looks like 6 cm/s is at about 90 to 100 km upstream of the sill, 0.78 at about 25 to 40 km upstream. 13 cm/s as needed for the 0.8deg layer estimate are not reached upstream of the sill...

# %%
v= b.v.swap_dims(dict(Y='dist'))
v_deep = b.v.where(b.th<0.7, drop=True).swap_dims(dict(Y='dist'))

# %%
if is_notebook:
    fig, ax = gv.plot.quickfig()
    v_deep.mean(dim=['Z']).swap_dims(dict(T='time')).plot()
    v_deep.mean(dim=['Z']).swap_dims(dict(T='time')).plot.contour(levels=[0.06, 0.078], colors='w', linewidths=0.75)
    ax.grid()

# %%
if is_notebook:
    for di in [-150, -100, -50, -30, -20]:
        v.isel(T=0).sel(dist=di, method='nearest').plot(y='Z')

    v_deep.isel(T=0).sel(dist=-30, method='nearest').plot(y='Z', color='k', linewidth=2)

# %% [markdown]
# ### bottom pressure from integrated density

# %% [markdown]
# Bottom pressure above was calculated from the *bottom pressure potential anomaly* by multiplying with density $\rho_0$.
#
# To compare with the twoyo data, we can calculate bottom pressure by integrating density from the surface and also by integrating density from somewhere around 4000 m depth.
#
# Actually, hydrostatic pressure anomaly $\phi^\prime$ is calculated as the integral over density plus the contribution of the free surface in the model:
# $$\phi^\prime = \frac{1}{\rho_c} \left( \rho_c g \eta + \int_z^0 (\rho - \rho_0) g \,dz \right)$$
# We multiply $\phi^\prime$ with $\rho_c$ to obtain $p^\prime$. In my code, $\rho_c$ is `rho0`. $\rho_c$ is `rhoConst` in `input/data`.

# %% [markdown]
# Here, we'll calculate $p_B^\prime$ as
# $$ p_B^\prime = \int_z^{4000m} \rho^\prime g dz $$

# %%
tmp = b.rhop.isel(Z=b.Z<-4100) * 9.81
pbp = -tmp.where(~np.isnan(tmp), other=0).integrate(coord='Z')

# %% [markdown]
# Compute the pressure due to the free surface elevation.

# %%
surface_p = nsl.model.gBaro * nsl.model.rho0 * b.eta

# %% [markdown]
# Compute baroclinic pressure for the full water column and plot for first and last time step of the analysis period.

# %%
bp = (nsl.model.gravity * b.rhop * b.drF).sum(dim='Z')

# %% [markdown]
# Compute bottom pressure due to non-hydrostatic pressure.

# %%
ii = b.HFacC.where(b.HFacC==1, other=100).argmax(dim='Z')
qbottom = b.q.isel(Z=ii-1)
q0bottom = b.q.isel(Z=ii-1).isel(T=0).swap_dims({'Y': 'dist'})
q1bottom = b.q.isel(Z=ii-1).isel(T=-1).swap_dims({'Y': 'dist'})

# %% [markdown]
# Plot components.

# %%
fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(6, 7), constrained_layout=True, sharex=True, sharey=True,
)
gv.plot.axstyle(ax1)
gv.plot.axstyle(ax2)


ti = 0
pbp.swap_dims({'Y': 'dist'}).isel(T=ti).plot(ax=ax1, label='bc $z<-4100\mathrm{m}$')
bp.swap_dims({'Y': 'dist'}).isel(T=ti).plot(ax=ax1, label='bc full depth')
(surface_p+bp).swap_dims({'Y': 'dist'}).isel(T=ti).plot(ax=ax1, label='bc + surface')
surface_p.swap_dims({'Y': 'dist'}).isel(T=ti).plot(ax=ax1, label='surface')
q0bottom.plot(ax=ax1, label='nh')
# b.swap_dims({'Y': 'dist'}).phibot.isel(T=ti).plot(ax=ax1, color='k', alpha=0.5, linestyle='--', label='full')
ax1.set(title='', xlabel='', ylabel=r'p$^\prime$ [N/m$^2$]')
gv.plot.annotate_corner(
    "model time 100 h",
    ax=ax1,
    quadrant=2,
    addx=-0.01,
    addy=-0.05,
)

ti = -1
pbp.swap_dims({'Y': 'dist'}).isel(T=ti).plot(ax=ax2, label='bc $z<-4100\mathrm{m}$')
bp.swap_dims({'Y': 'dist'}).isel(T=ti).plot(ax=ax2, label='bc full depth')
(surface_p+bp).swap_dims({'Y': 'dist'}).isel(T=ti).plot(ax=ax2, label='bc + surface')
surface_p.swap_dims({'Y': 'dist'}).isel(T=ti).plot(ax=ax2, label='surface')
q1bottom.plot(ax=ax2, label='nh')
# b.swap_dims({'Y': 'dist'}).phibot.isel(T=ti).plot(ax=ax2, color='k', alpha=0.5, linestyle='--', label='full')

ax2.set(xlim=(-40, 100), title='', ylabel=r'p$^\prime$ [N/m$^2$]', xlabel='Distance [km]')
ax2.legend(loc='lower left', bbox_to_anchor=(0.65, 0.62));
gv.plot.annotate_corner(
    "model time 150 h",
    ax=ax2,
    quadrant=2,
    addx=-0.01,
    addy=-0.05,
)
nsl.io.save_png('model_pressure')

for axi in [ax1, ax2]:
    axi.grid(
        which="major",
        axis="both",
        color="0.5",
        linewidth=0.1,
        linestyle="-",
        alpha=0.3,
    )

nsl.io.save_pdf('model_pressure')

# %% [markdown]
# ## Save model bottom pressure

# %%
bottom_pressure = xr.Dataset(data_vars={'bottom': pbp, 'full': b.phibot, 'surface': surface_p, 'bcfull': bp, 'nh': qbottom})

# %%
bottom_pressure

# %%
cfg.model.output.bottom_pressure

# %%
save_name = cfg.model.output.bottom_pressure
print('saving model bottom pressure to {}'.format(save_name))
bottom_pressure.to_netcdf(save_name)
