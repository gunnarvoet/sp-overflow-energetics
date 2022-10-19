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
# # Plot Model Snapshot

# %%
cfg = nsl.io.load_config()

# %% [markdown]
# ## Read model data

# %%
b = xr.open_dataset(cfg.model.output.data)

# %%
print('start: ', b.time[0].data, '\nend  : ', b.time[-1].data)

# %% [markdown]
# ## Plot Snapshot

# %%
nsl.model.plot_snapshot(b, ti=0, zlim=3600)
nsl.io.save_png('model_snapshot')
nsl.io.save_pdf('model_snapshot')
