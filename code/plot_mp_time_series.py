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
import matplotlib.pyplot as plt
import numpy as np

import gvpy as gv
import nslib as nsl

# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'
# %autosave 300

# %% [markdown]
# # Plot MP T1 Velocity Time Series

# %%
# Load moored profiler data
mp = nsl.io.load_mp_t1()

# %%
V, lpdata, bpdata =nsl.plt.PlotMooredProfilerTimeSeries(mp)
nsl.io.save_png('T1_time_series')
nsl.io.save_pdf('T1_time_series')

# %% [markdown]
# Calculate mean kinetic energy for low-frequency and tidal flow.

# %%
fig, ax = gv.plot.quickfig()
mlpv = np.mean(lpdata, axis=0)
ax.plot(mlpv, -V.depth)

# %%
# pick only the dense layer (take the middle of the shear layer )
mask = V.depth > 4300

# %%
dz = np.sum(np.diff(V.depth[mask]))


# %%
def mean_ke(data):
    rho0 = 1e3
    ke = dz * 0.5 * rho0 * data ** 2
    ke_mean = np.nanmean(ke[:, mask])
    return ke_mean


# %% [markdown]
# Units here are J/m^3. Maybe integrate in the vertical to have similar units as in the energy budget... units are then J/m^2.

# %%
print('-------------------')
print('--- Upstream KE ---')
print('-------------------\n')

lp_ke_m = mean_ke(lpdata)
print(f'low-frequency band KE: {lp_ke_m/1e3:1.1f} kJ/m^2')

bp_ke_m = mean_ke(bpdata)
print(f'tidal band KE: {bp_ke_m/1e3:1.1f} kJ/m^2')

print(f'ratio of low-frequency KE to tidal KE is {lp_ke_m / bp_ke_m:1.1f}')

print('\n-------------------\n')
