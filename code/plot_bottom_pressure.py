#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Plot Bottom Pressure

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
nsl.plt.PlotFormDrag()
nsl.io.save_png('bottom_pressure')

# %%
nsl.plt.PlotFormDrag(for_pdf=True)
nsl.io.save_pdf('bottom_pressure')

# %%
