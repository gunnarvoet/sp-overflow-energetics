#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Plot Energetics Fields Integrated
# for towyos and model

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
# Read towyo data, including energetics calcs
a = nsl.io.load_towyos()

# %%
# Read model data
b = xr.open_dataset(cfg.model.output.snapshot)
E = xr.open_dataset(cfg.model.output.energy_snapshot)
E.coords['depth'] = (['Z'], -1*E.Z.data)
b.coords['depth'] = (['Z'], -1*b.Z.data)
E = E.swap_dims({'Z': 'depth', 'Y': 'dist'})
b = b.swap_dims({'Z': 'depth', 'Y': 'dist'})
# Calculate model energy fluxes
E['Ekpflux'] = E.Ekp * b.v
E['Ekpflux2dabs'] = np.sqrt((E.Ekp * b.v)**2 + (E.Ekp * b.W)**2)
E['Eppflux'] = E.Epp_sorted_rho * b.v

# %%
fig, ax = gv.plot.quickfig(h=2)
ty = a['t12']
b.Depth.plot()
ty.topo.plot(x='dist')
ax.set(ylim=(5300, 4600), xlim=(-20, 40));

# %% [markdown]
# Plot integrated energetics terms.

# %%
print('plotting integrated energetics terms')
nsl.plt.PlotEnergeticsTermsIntegrated(a, b, E, model_int_limit_isot=[0.9, 0.8])
nsl.io.save_png('energetics_fields_integrated')

# %%
nsl.plt.PlotEnergeticsTermsIntegrated(a, b, E, model_int_limit_isot=[0.9, 0.8], for_pdf=True)
nsl.io.save_pdf('energetics_fields_integrated')
