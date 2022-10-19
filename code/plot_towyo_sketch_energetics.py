#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# #### Imports

# %%
import xarray as xr
import nslib as nsl

# %config InlineBackend.figure_format = 'retina'
# %% [markdown]
# # Plot Towyo Energetics Sketch
# %% [markdown]
# Plot a sketch showing terms of the energy budget.

# %%
cfg = nsl.io.load_config()

# %%
nsl.plt.PlotTowyoSketch()
nsl.io.save_png("towyo_sketch_energetics")
nsl.io.save_pdf("towyo_sketch_energetics")
