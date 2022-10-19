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
# # Extract Model Snapshot
# Save a snapshot from the model time series including energetics for plotting.

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
# Read config

# %%
cfg = nsl.io.load_config()

# %% [markdown]
# Read model data

# %%
b = xr.open_dataset(cfg.model.output.data)

# %%
E = xr.open_dataset(cfg.model.output.energy)

# %% [markdown]
# Save snapshots

# %%
# select time step
ti = cfg.parameters.model.snapshot_ind

# %%
nsl.io.close_nc(cfg.model.output.energy_snapshot)
E.isel(T=ti).to_netcdf(cfg.model.output.energy_snapshot)

# %%
nsl.io.close_nc(cfg.model.output.snapshot)
b.isel(T=ti).to_netcdf(cfg.model.output.snapshot)
