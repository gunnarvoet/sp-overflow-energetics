#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Plot Energy Budget Results

# %% [markdown]
# Both for towyos and model.

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import gvpy as gv
import nslib as nsl
# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'
# %autosave 300

# %%
cfg = nsl.io.load_config()

# %%
nsl.plt.PlotEnergeticsLayerBudgetResultsHorizontalTwoIsothermsSourcePositive(
    cfg, refrho_sorted=True
)
nsl.io.save_png("energy_budget_results")
nsl.io.save_pdf("energy_budget_results")
