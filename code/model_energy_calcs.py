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
# # Model Energy Calcs

# %% [markdown]
# Calculate energy terms for the Northern Sill model data. This is not the energy budget yet.

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import xarray as xr
import gvpy as gv
import nslib as nsl

# %config InlineBackend.figure_format = 'retina'
# %reload_ext autoreload
# %autoreload 2
# %autosave 300


# %% [markdown]
# Load config

# %%
cfg = nsl.io.load_config()

# %% [markdown]
# Load model data

# %%
b = xr.open_dataset(cfg.model.output.data)

# %% [markdown]
# ## Calculate energy terms

# %% [markdown]
# Here we use the initial downstream density profile to define APE and $p^\prime$.

# %% [markdown]
# Now also including the calculation of APE and horizontal internal wave fluxes based on sorted initial stratification and a mean density profile thereof.

# %% [markdown]
# Calculate the energy terms

# %%
print('calculate energetics')
E = nsl.model.calculate_energetics(b)

# %% [markdown]
# Add other types of internal wave fluxes to the dataset. These are calculated in `model_wave_flux_calcs.py`. They need to be interpolated the the model grid as we calculated these for a smaller region centered around the sill.

# %% [markdown]
# Internal wave fluxes based on local mean profiles

# %%
lp_fluxes = xr.open_dataset(cfg.model.output.small_scale_fluxes).swap_dims({'dist': 'Y', 'time': 'T'})

# %%
lp_fluxes_i = lp_fluxes.interp_like(E)

# %%
E['lp_hwf'] = lp_fluxes_i.hwf
E['lp_vwf'] = lp_fluxes_i.vwf

# %% [markdown]
# Internal wave fluxes based on high-pass filtered time series.

# %%
hp_fluxes = xr.open_dataset(cfg.model.output.hp_fluxes).swap_dims({'dist': 'Y', 'time': 'T'})

# %%
hp_fluxes_i = hp_fluxes.interp_like(E)

# %%
E['hp_hwf'] = hp_fluxes_i.hwf
E['hp_vwf'] = hp_fluxes_i.vwf

# %% [markdown]
# Save

# %%
print('saving energy terms to', cfg.model.output.energy)

# %%
nsl.io.close_nc(cfg.model.output.energy)
E.to_netcdf(cfg.model.output.energy)
