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
# # Model Energy Budget

# %% [markdown]
# ## Read model data

# %%
cfg = nsl.io.load_config()

# %%
b = xr.open_dataset(cfg.model.output.data)

# %%
print("start: ", b.time[0].data, "\nend  : ", b.time[-1].data)

# %%
E = xr.open_dataset(cfg.model.output.energy)

# %% [markdown]
# ## Energy budget of a layer

# %% [markdown]
# Similar to the energy budget in a box below but with an upper isotherm as boundary instead of rigid fixed depth.

# %% [markdown]
# ### Play with integration range

# %%
b1 = b.isel(T=range(100))

# %%
E1 = E.isel(T=range(100))

# %%
config2 = cfg.copy()

# %%
cfg.parameters.model.integration_horizontal_min = 0

# %%
cfg.parameters.model.integration_horizontal_max = 25

# %%
B1 = nsl.model.energy_budget_layer(cfg, b1, E1, isot=0.8)

# %%
Epp = "EppSortedFluxDivergence"
IWVert = "IWSortedFluxVertDivergence"
IWHoriz = "IWSortedFluxHorizDivergence"
Epp = "EppFluxDivergence"
IWVert = "IWFluxVertDivergence"
IWHoriz = "IWFluxHorizDivergence"
plot_terms = [
    "Unsteadiness",
    Epp,
    "EkpFluxDivergence",
    IWHoriz,
    IWVert,
    "IWFluxVertDivergence_hp",
    "Dissipation",
    "BottomDrag",
]
means = []
stds = []
for var in plot_terms:
    means.append(B1[var].mean(dim="T").data)
    stds.append(B1[var].std(dim="T").data)
means = np.array(means)
stds = np.array(stds)
# print layer results
print("-----")
print("budget results for layer [kW/m]:")
for name, data, sigma in zip(plot_terms, means, stds):
    print(name, f"{data/1e3:1.2f} +- {sigma/1e3:1.2f}")
sum_terms = means[[1, 2, 3, 4, 6, 7]]
print("terms for residuals:", sum_terms)
residual = np.sum(means[[1, 2, 3, 4, 6, 7]])
print(f"residual: {residual/1e3:1.2f}")
sigmas_squared = np.array(stds[[1, 2, 3, 4, 6, 7]]) ** 2
combined_error = np.sqrt(np.sum(sigmas_squared))
print(f"residual sigma: {combined_error/1e3:1.2f}")
print("-----")

# %% [markdown]
# Save energy budget results.

# %%
nsl.io.close_nc(cfg.model.output.energy_budget_results_layer_08_25)

# %%
B1.to_netcdf(cfg.model.output.energy_budget_results_layer_08_25)

# %% [markdown]
# Compare sorted density method with downstream reference profile method. This reduces the overall amount of energy available by about a factor of 2.

# %%
if is_notebook:
    B1.EppSortedFluxDivergence.plot(color='C3', linestyle='-', label="APE sorted")
    B1.EppFluxDivergence.plot(color='C3', linestyle='--', label="APE")
    B1.IWFluxHorizDivergence.plot(color='C0', linestyle='--', label="IW flux")
    B1.IWSortedFluxHorizDivergence.plot(color='C0', linestyle='-', label="IW flux sorted")
    plt.legend()
    print(
        f"sorted: {(B1.EppSortedFluxDivergence + B1.IWSortedFluxHorizDivergence).mean().data:12.2}"
    )
    print(
        f"downstream: {(B1.EppFluxDivergence + B1.IWFluxHorizDivergence).mean().data:7.2}"
    )

# %% [markdown]
# How does the vertical internal flux differ between the two methods?

# %%
if is_notebook:
    B1.IWSortedFluxVertDivergence.plot()
    B1.IWFluxVertDivergence.plot()
    print(
        f"sorted: {B1.IWSortedFluxVertDivergence.mean().data:12.2}"
    )
    print(
        f"downstream: {B1.IWFluxVertDivergence.mean().data:8.2}"
    )

# %%
if is_notebook:
    nsl.model.plot_energy_budget(B1, plotall=True)

# %% [markdown]
# ### Regular integration

# %%
cfg = nsl.io.load_config()

# %%
B = nsl.model.energy_budget_layer(cfg, b, E, isot=0.9)

# %% [markdown]
# Save energy budget results.

# %%
nsl.io.close_nc(cfg.model.output.energy_budget_results_layer)

# %%
B.to_netcdf(cfg.model.output.energy_budget_results_layer)

# %% [markdown]
# Plot all terms. Internal wave fluxes shown here are based on a downstream reference profile.

# %%
if is_notebook:
    nsl.model.plot_energy_budget(B, plotall=True)

# %% [markdown]
# ### Integrate only to 0.8deg isotherm

# %%
cfg = nsl.io.load_config()

# %%
B08 = nsl.model.energy_budget_layer(cfg, b, E, isot=0.8)

# %% [markdown]
# Save energy budget results.

# %%
nsl.io.close_nc(cfg.model.output.energy_budget_results_layer_08)

# %%
B08.to_netcdf(cfg.model.output.energy_budget_results_layer_08)

# %% [markdown]
# ## Energy budget in a box

# %% [markdown]
# Function spits out the various terms for the baroclinic energy equation integrated over a box of variable width and full water column.

# %%
if is_notebook:
    # a quick look at topography
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,2))
    ax.fill_between(b.dist,b.Depth, 1e4, color='k')
    ax.contour(b.dist, -b.Z, b.isel(T=-1).th, np.arange(0.5, 1.4, 0.05), colors='0.2',
               linewidths=0.5)
    nsl.model.plot_center(dy=60, zz=3000)

# %%
Bbox = nsl.model.energy_budget_box(cfg, b, E, zlim=-4200)

# %%
nsl.io.close_nc(cfg.model.output.energy_budget_results_box)

# %%
Bbox.to_netcdf(cfg.model.output.energy_budget_results_box)

# %% [markdown]
# Since BT/BC-Conversion is basically zero, the sum of all baroclinic energy terms should be zero as well. I can't seem to figure out what is wrong here!

# %%
if is_notebook:
    nsl.model.plot_energy_budget(Bbox, plotall=True)
# plt.savefig('fig/model_jesse_box_budget_z4400.png', dpi=150, bbox_inches='tight')

# %%
if is_notebook:
    Bbox.EppSortedFluxDivergence.plot(color='C3', linestyle='-', label="APE sorted")
    Bbox.EppFluxDivergence.plot(color='C3', linestyle='--', label="APE")
    Bbox.IWFluxHorizDivergence.plot(color='C0', linestyle='--', label="IW flux")
    Bbox.IWSortedFluxHorizDivergence.plot(color='C0', linestyle='-', label="IW flux sorted")
    plt.legend()
    print(
        f"sorted: {(Bbox.EppSortedFluxDivergence + Bbox.IWSortedFluxHorizDivergence).mean().data:12.2}"
    )
    print(
        f"downstream: {(Bbox.EppFluxDivergence + Bbox.IWFluxHorizDivergence).mean().data:7.2}"
    )

# %% [markdown]
# ## Evolution of KE, APE over the whole domain

# %%
if is_notebook:
    allEK = (E.Ekp * b.drF * b.dyF * b.HFacC).sum(dim=('Z', 'Y'))
    allAPE = (E.Epp * b.drF * b.dyF * b.HFacC).sum(dim=('Z', 'Y'))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax.plot(b.time, allEK, label='KE')
    ax.plot(b.time, allAPE, label='APE')
    ax.legend()
    ax.grid(True)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax.plot(b.time, np.gradient(allAPE+allEK, b.time*3600))
