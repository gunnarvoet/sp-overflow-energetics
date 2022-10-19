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

# %% [markdown]
# # Plot Model Steadiness
# Plot steadiness terms in Northern Sill model run

# %%
cfg = nsl.io.load_config()

# %%
cfg.model.input.full_model_run

# %% [markdown]
# ## Read model data

# %% [markdown]
# We need to read the whole model run to show spinup, instability towards the end.

# %%
b = nsl.model.load()

# %% [markdown]
# Subsample

# %%
trange = range(0, 1140, 20)

# %%
b = b.isel(T=trange)

# %% [markdown]
# Set an identifier for this run. This is used to save/reload APE calculation results which take a while to run.

# %%
b.attrs['modelrun'] = 'full'

# %% [markdown]
# Basic variables

# %%
b = nsl.model.calculate_density(b)

# %%
b = nsl.model.calculate_pressure(b)

# %%
b = nsl.model.calculate_pressure_anomaly_sorted_rho(b, cfg)

# %% [markdown]
# Barotropic/Baroclinic Velocities

# %%
b = nsl.model.calculate_velocities(b)

# %% [markdown]
# Energetics

# %%
E = nsl.model.calculate_energetics(b)

# %% [markdown]
# Energy budget of a layer

# %%
# change integration limits to be a bit outside our
# energy budget to capture a wider region
cfgtmp = cfg.copy()
cfgtmp.parameters.model.integration_horizontal_max = 30
cfgtmp.parameters.model.integration_horizontal_min = -10

# %%
B = nsl.model.energy_budget_layer(cfgtmp, b, E, isot=0.9, steadiness_terms_only=True)

# %%
nsl.model.plot_steadiness_terms(B)
nsl.io.save_png('model_steadiness')
nsl.io.save_pdf('model_steadiness')
