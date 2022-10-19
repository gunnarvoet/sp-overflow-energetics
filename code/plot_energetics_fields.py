#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Plot Energetics Fields
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

# %%
# Read model data
b = xr.open_dataset(cfg.model.output.snapshot)
E = xr.open_dataset(cfg.model.output.energy_snapshot)

# %%
# Adjust coordinates
E.coords['depth'] = (['Z'], -1*E.Z.data)
b.coords['depth'] = (['Z'], -1*b.Z.data)
E = E.swap_dims({'Z': 'depth', 'Y': 'dist'})
b = b.swap_dims({'Z': 'depth', 'Y': 'dist'})

# %%
# Calculate model energy fluxes
E['Ekpflux'] = E.Ekp * b.v
E['Ekpflux2dabs'] = np.sqrt((E.Ekp * b.v)**2 + (E.Ekp * b.W)**2)
E['Eppflux'] = E.Epp_sorted_rho * b.v

# %%
# Read model fluxes based on local spatial mean profiles
fluxes = xr.open_dataset(cfg.model.output.small_scale_fluxes)
fluxes.coords['depth'] = (['Z'], -1*fluxes.Z.data)
fluxes = fluxes.swap_dims({'Z': 'depth'})

# %% [markdown]
# Plot energetics terms.

# %%
print("plotting energetics terms")
nsl.plt.PlotEnergeticsTerms(a, b, E, fluxes, cfg)
nsl.io.save_png("energetics_fields")
nsl.io.save_pdf("energetics_fields")

