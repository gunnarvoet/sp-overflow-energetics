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

# %% [markdown] heading_collapsed=true
# #### Imports

# %% hidden=true
# %matplotlib inline
from pathlib import Path
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import gsw
import pickle
import scipy.signal as sg

from tqdm.notebook import tqdm

import gvpy as gv
import nslib as nsl
from nslib import model_wave_fluxes as mwf

# %config InlineBackend.figure_format = 'retina'

# %reload_ext autoreload
# %autoreload 2
# %autosave 300


# %% hidden=true
gv.misc.warnless()

# %% [markdown]
# # Model Wave Flux Calcs

# %% [markdown]
# Calculate primed quantities and internal wave fluxes as for the observations by determining local mean profiles of density, velocity. Local here means within $\pm$2.5$\,$km.

# %% [markdown]
# In addition, calculate model internal wave fluxes based on high-pass filtered model time series.

# %%
cfg = nsl.io.load_config()

# %%
print('calculate internal wave fluxes based on local spatial mean profiles')
mwf.small_scale_flux_calculations()

# %%
print('calculate high-pass filter-based internal wave fluxes')
mwf.hp_flux_calculations()
