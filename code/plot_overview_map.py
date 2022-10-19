#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Plot Overview Map

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import gvpy as gv
import nslib as nsl
import overview_map

# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'
# %autosave 300

# %%
om = overview_map.OverviewMap()

# %%
om.shading_calcs()

# %%
# activate to plot individual panels for diagnostics
if 0:
    om.plot_globe()
    om.plot_samoan_passage()
    om.plot_northern_sill()

# %%
om.plot_overview_map()
nsl.io.save_png('map_overview')
nsl.io.save_pdf('map_overview')
