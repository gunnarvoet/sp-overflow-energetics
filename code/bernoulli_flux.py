#!/usr/bin/env python
# coding: utf-8
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
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import xarray as xr
import gvpy as gv
import nslib as nsl
from pathlib import Path

import nslib.bernoulli as bern

# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'
# %autosave 300

# %%
cfg = nsl.io.load_config()

# %%
is_notebook = gv.misc.is_notebook()

# %% [markdown]
# #### Load Data

# %% [markdown]
# Towyos.

# %%
a = nsl.io.load_towyos()

# %% [markdown]
# Model hours 100 to 150.

# %%
b = xr.open_dataset(cfg.model.output.data)

# %% [markdown]
# ## Notes

# %% [markdown]
# - It seems like for just a snapshot the Bernoulli flux is very messy. In the model we can average over a bunch of time steps and then have a pretty clear drop over the first hydraulic jump. In the observations we can't do this. Maybe this is then a good motivation for the more detailed energy equation and its application to the problem?
# - We could actually augment the observations with data from the moored profilers and calculate the Bernoulli function at these points as well. This might give us a better idea of the drop across the hydraulic jump. I am just not sure if I would like to add another data source to the paper...

# %% [markdown]
# ## 1.5 Layer Energy / Bernoulli flux

# %% [markdown]
# While we compute a more thorough and detailed energy budget of the flow elsewhere, we also look at a simple 1.5 layer model type Bernoulli flux calculation as a sanity check.

# %% [markdown]
# The Bernoulli flux may be formulated with or without the presence of entrainment. The Bernoulli function is defined everywhere except for regions of hydraulic jumps. The value of the Bernoulli function may either change due to entrainment (is this true?) or due to energy loss.

# %% [markdown]
# ---

# %% [markdown]
# Larry's notes on form drag contain the 1.5 layer formulation of the Bernoulli function drop across the hydraulic jump that Jody also calculates in his notes.
# Neglecting any entrainment and thus keeping the volume transport constant, Larry starts out with the steady shallow-water equations
#
# $$v \frac{\partial v}{\partial y} + g^\prime \frac{\partial d}{\partial y} = - g^\prime \frac{\partial h}{\partial y}$$
#
# $$\frac{\partial (vd)}{\partial y} = 0$$
#
# where $d$ is the thickness of the layer and $h$ is the elevation of the topography.
# Integration of the second equation shows that the volume transport per unit width $Q=vd$ is constant which also holds across a hydraulic jump as there is no entrainment. Integrating the first equation results in the Bernoulli equation describing the energy of the system which is constant everywhere except for the region of the hydraulic jump.
#
# $$B = \frac{v^2}{2} + g^\prime d + g^\prime h$$
#
# The value of $B$ drops from upstream of the jump to downstream. The change in the energy flux is
#
# $$-\Delta E = Q_u B_u -Q_d B_d = Q(B_u - B_d)$$ 
#
# $$-\Delta E = v_u d_u (\frac{v_u^2}{2} + g^\prime d_u - \frac{v_d^2}{2} -g^\prime d_d)\ .$$
#
# If entrainment is allowed then the volume flux changes and the drop is 
#
# $$-\Delta E = v_u d_u (\frac{v_u^2}{2} + g^\prime d_u) - v_d d_d(\frac{v_d^2}{2} -g^\prime d_d)\ .$$
#
# Note that here the bottom depth is the same upstream and downstream of the sill. Jody derives the energy drop in his notes with a change in bottom depth which adds the term with $h$:
#
# $$-\Delta E = \frac{v_u^3 d_u}{2} + v_u g^\prime (d_u^2 + h_u d_u) - \frac{v_d^3 d_d}{2} -v_d g^\prime (d_d^2 + h_d d_d)\ .$$
#
# We can calculate this for various points upstream and downstream of the sill relatively easily. There is the issue of defining the density anomaly $g^\prime$ and the layer velocity $v$. The layer velocity could be either the mean or median velocity calculated over the depth of the layer. Density anomaly will be calculated against a density somewhere above the interface. The calculations can be done both for the observations and for the model.

# %% [markdown]
# ### New Implementation

# %% [markdown]
# I have now defined classes for the Bernoulli function calculation with both model and towyo data. They make the calculation of Bernoulli function transport much more straight forward than my earlier code (still kept below).

# %% [markdown]
# Run for a single model time step.

# %%
cs = b.isel(T=50).where(b.Z<-3000, drop=True).swap_dims({'Y': 'dist'})

# %%
Ms = bern.ModelBernoulli(cs, interface=0.8)

# %% [markdown]
# Let's find the deepest topography in the model and use this as a reference depth to calculate bottom elevation in both towyos and observations.

# %%
Ms.data.Depth.isel(Z=0).sel(dist=200, method='nearest').round().item()

# %%
if is_notebook:
    fig, ax = gv.plot.quickfig()
    Ms.B.plot()
    Ms.B.rolling(dist=100, center=True).mean().plot()
    ax.set(xlim=(-10, 50), ylim=(20, 60), title='')

# %% [markdown]
# Calculate $B$ for both towyos.

# %%
T1 = bern.TowyoBernoulli(a['t12'], interface=45.94)
T2 = bern.TowyoBernoulli(a['t14'], interface=45.94)

# %% [markdown]
# Calculate $B$ for several time steps in the model. We skip every other time step to speed up the calculation a bit.

# %%
c = b.isel(T=np.arange(0,200,2)).where(b.Z<-3000, drop=True).swap_dims({'Y': 'dist'})

# %%
M = bern.ModelBernoulli(c, interface=0.8)

# %%
if is_notebook:
    fig, ax = gv.plot.quickfig()
    M.B.plot(y='T', robust=True, cbar_kwargs=dict(shrink=0.7, aspect=30, label='$v\,dB$ [kW/m]'))
    ax.set(xlim=(-50, 90), title='');

# %% [markdown]
# Plot results from towyos, model snapshot, and model mean.

# %%
alpha_raw = 0.3
colors = ["#0277BD", "#6A1B9A", "C6"]
fig, ax = gv.plot.quickfig(w=5, h=4)
model_smoothed_mean = M.B.mean(dim="T").rolling(dist=100, center=True).mean()
hstd = ax.fill_between(
    M.B.dist,
    model_smoothed_mean + 2 * M.B.std(dim="T"),
    model_smoothed_mean - 2 * M.B.std(dim="T"),
    fc="0.5",
    alpha=0.3,
    ec=None,
)
lw_thin = 1
lw_thick = 1.5
hmean = model_smoothed_mean.plot(color='0.5', linewidth=lw_thick, label='model mean')
Ms.B.plot(color=colors[2], linewidth=lw_thin, alpha=alpha_raw)
Ms.B.rolling(dist=100, center=True).mean().plot(color=colors[2], linewidth=lw_thick, label='model snapshot')
T1.B.plot(color=colors[0], linewidth=lw_thin, alpha=alpha_raw)
T1.B.rolling(dist=10, center=True).mean().plot(color=colors[0], linewidth=lw_thick, label='towyo 2012')
T2.B.plot(color=colors[1], linewidth=lw_thin, alpha=alpha_raw)
T2.B.rolling(dist=10, center=True).mean().plot(color=colors[1], linewidth=lw_thick, label='towyo 2014')
ax.set(
    xlim=(-4, 44),
    ylim=(0, 75),
#     title="Bernoulli function transport",
    ylabel=r"$\rho_0\,Q\,B$ [kW/m]",
    xlabel='distance [km]',
)
ax.legend()
nsl.io.save_png('bernoulli_transport')
nsl.io.save_pdf('bernoulli_transport')

# %% [markdown]
# Calculate the model drop between kilometer 0 and 17 based on the smoothed model mean.

# %%
model_smoothed_mean.sel(dist=0, method='nearest').item() - model_smoothed_mean.sel(dist=17, method='nearest').item()

# %% [markdown]
# How does the drop look like if we go further into the far field?

# %%
fig, ax = gv.plot.quickfig()
model_smoothed_mean.plot()
ax.set(xlim=(-10, 150))

# %%
model_smoothed_mean.sel(dist=0, method='nearest').item() - model_smoothed_mean.sel(dist=80, method='nearest').item()

# %% [markdown]
# Calculate form drag based on Larry's parameterization:
#
# $$ \rho \int_z\int_y\, \epsilon\, dy\, dz = 0.9744 v_a D_f + 0.8608 \frac{v_a}{\rho g' d_a^2} D_f^2$$
#
# where $v_a$ and $d_a$ are upstream velocity and layer thickness.

# %%
da = T1.d.where(T1.data.dist<2).mean().item()

# %%
va = T1.v.where(T1.data.dist<2).mean().item()

# %%
gpr = T1.gprime.where(T1.data.dist<2).mean().item()

# %%
rho = T1.rho0

# %% [markdown]
# 2012 form drag in N/m

# %%
Df_2012 = -3.1e4
Df_2014 = -3.6e4

# %%
Df_model = -1.6e4

# %%
dissipation_estimate = 0.9744 * va * Df_2012 + 0.8608 * va * Df_2012**2 / (rho * gpr * da**2)

# %%
dissipation_estimate

# %%
T1 = bern.TowyoBernoulli(a['t12'], interface=45.94)
T2 = bern.TowyoBernoulli(a['t14'], interface=45.94)

# %%
T1.calculate_single_layer_form_drag_dissipation(Df=Df_2012)

# %%
T2.calculate_single_layer_form_drag_dissipation(Df=Df_2014)

# %% [markdown]
# Form drag dissipation in model snapshot:

# %%
Ms.calculate_single_layer_form_drag_dissipation(Df=Df_model)

# %% [markdown]
# Model form drag dissipation time series:

# %%
model_form_drag_dissipation = M.calculate_single_layer_form_drag_dissipation(Df=Df_model)

# %%
print(np.mean(model_form_drag_dissipation))

# %%
fig, ax = gv.plot.quickfig()
ax.plot(M.data.time, model_form_drag_dissipation)
