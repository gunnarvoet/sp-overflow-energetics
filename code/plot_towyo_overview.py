# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python [conda env:sp]
#     language: python
#     name: conda-env-sp-py
# ---

# %%
# %matplotlib inline
import cmocean
import gsw
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

import gvpy as gv
import nslib as nsl

# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'

# %%
is_notebook = gv.misc.is_notebook()

# %%
col, col2 = nsl.plt.load_colors()

# %% [markdown]
# # Plot Towyo Overview

# %% [markdown]
# Load towyo data

# %%
a = nsl.io.load_towyos()


# %% [markdown]
# Shear

# %%
def FilterVelocity(ty,Wn):
    import scipy.signal as sg
    vlp = np.ones_like(ty.v)*np.nan
    v = ty.v.values
    N  = 3    # Filter order
    B, A = sg.butter(N, Wn, output='ba')
    for i,vi in enumerate(v.T):
        vlp[np.where(~np.isnan(vi)),i] = sg.filtfilt(B,A,vi[~np.isnan(vi)])
    return vlp


# %% [markdown]
# Extract TPXO barotropic tide prediction along the two towyos. Save them so we don't have to have pytide installed.

# %%
def extract_tpxo(ty):
    print("extracting TPXO model data")
    t2 = pd.to_datetime(ty.time.values)
    yday = t2.dayofyear+t2.hour/24+t2.minute/24/60+t2.second/24/3600
    year = t2.year
    from pytide import model
    tidemod = model('tpxo7.2')
    velu = []
    velv = []
    h = []
    for yy, yd, lon, lat in zip(year, yday, ty.lon.values, ty.lat.values):
        vel = tidemod.velocity(yy, yd, lon, lat)
        velu.append(vel.u.data)
        velv.append(vel.v.data)
        htmp = tidemod.height(yy, yd, lon, lat)
        h.append(htmp.h.data)
    velu = np.concatenate(velu)
    velv = np.concatenate(velv)
    h = np.concatenate(h)
    ty['tpxo_velv'] = (['x'], velv)
    ty['tpxo_velu'] = (['x'], velu)
    ty['tpxo_h'] = (['x'], h)
    return ty


# %%
def save_tpxo(ty):
    cfg = nsl.io.load_config()
    tpxo = xr.merge([ty.tpxo_velu, ty.tpxo_velv, ty.tpxo_h])
    savename = f"tpxo_{ty.name}.nc"
    savepath = cfg.path.output.joinpath(savename)
    tpxo.to_netcdf(savepath)


# %%
def read_tpxo(ty):
    cfg = nsl.io.load_config()
    loadname = f"tpxo_{ty.name}.nc"
    loadpath = cfg.path.output.joinpath(loadname)
    tpxo = xr.open_dataset(loadpath)
    ty['tpxo_velv'] = (['x'], tpxo.tpxo_velv.data)
    ty['tpxo_velu'] = (['x'], tpxo.tpxo_velu.data)
    ty['tpxo_h'] = (['x'], tpxo.tpxo_h.data)


# %%
for k, ty in a.items():
    cfg = nsl.io.load_config()
    loadname = f"tpxo_{ty.name}.nc"
    loadpath = cfg.path.output.joinpath(loadname)
    if loadpath.exists() is False:
        ty = extract_tpxo(ty)
        save_tpxo(ty)
    else:
        read_tpxo(ty)


# %% [markdown]
# # Figure

# %%
# plotting functions for fig 1
def Fig1Shear(ty,ax):
    """Plot shear
    
    Args:
        ty (xarray Dataset): towyo dataset
        ax (ax obj): handle to axis
        
    Returns:
        h (plot object): handle to contour (theta) plot
        ht (plot object): handle to topo plot
    """
    print('plotting shear')
    
    vlp = FilterVelocity(ty, 1/50)
    dv = np.gradient(vlp, axis=0)
    
    h = ax.contourf(ty.dist,ty.z,dv**2,
                      levels=np.arange(0, 1e-5, 1e-6),
                      cmap='BuPu', antialiased=True, extend='max')
    for c in h.collections:
        c.set_rasterized(True)
        c.set_edgecolor("face")
    
    ax.contour(ty.dist, ty.z, ty.gsw_sigma4, levels=ty.sg4bins, colors='0.5', linewidths=0.5)
    ax.contour(ty.dist, ty.z, ty.gsw_sigma4, levels=[45.94], colors='k', linewidths=0.75)
    
    ht = ax.plot(ty.dist,ty.topo,'k')
    ax.set_xlabel('Distance [km]')
    ax.set_ylim([5200,4050])
    ax.set_xlim([-1,32])
    ax.set_ylabel('Depth [m]')
    return h


# %%
if is_notebook:
    fig, ax = gv.plot.newfig(8,5)
    ty = a['t14']
    Fig1Shear(ty,ax);


# %%
# plotting functions for fig 1
def Fig1THw(ty,wvar,ax):
    """Plot potential temperature and LADCPw
    
    Args:
        ty (xarray Dataset): towyo dataset
        wvar (str): Variable for vertical velocity
        ax (ax obj): handle to axis
        
    Returns:
        h (plot object): handle to contour (theta) plot
        h2 (plot object): handle to quiver (w) plot
        qk (plot object): handle to quiver key
        ht (plot object): handle to topo plot
    """
    print('plotting theta and w...')
    h = ax.contourf(ty.dist,ty.z,ty.th,
                      levels=np.arange(0.675,1.125,0.025),
                      cmap='Spectral_r')
    for c in h.collections:
        c.set_rasterized(True)
        c.set_edgecolor("face")
    h2 = ax.quiver(ty.dist,ty.z[::20],np.zeros_like(ty[wvar][::20,:]),
                   ty[wvar][::20,:],np.sign(ty[wvar][::20,:]),
                   cmap='Greys',
                   units='y',
                   scale=-1e-3)
    qk = ax.quiverkey(h2, 2, 5100, 0.05, '5 cm/s',
                   labelpos='E',
                   coordinates='data',
                   fontproperties={'weight': 'normal'})
    ht = ax.plot(ty.dist,ty.topo,'k')
    ax.set_xlabel('Distance [km]')
    ax.set_ylim([5200,4050])
    ax.set_xlim([-1,32])
    ax.set_ylabel('Depth [m]')
    return h, h2, qk, ht


# %%
if is_notebook:
    fig, ax = gv.plot.newfig(8,5)
    ty = a['t12']
    Fig1THw(ty,'w',ax);


# %%
# plot epsilon from overturns
def Fig1Eps(ty,ax):
    """Plot epsilon
    
    Args:
        ty (xarray Dataset): towyo dataset
        ax (ax obj): handle to axis
        
    Returns:
        h (plot object): handle to pcolormesh (eps) plot
        ht (plot object): handle to topo plot
        """
    print('plotting epsilon...')
    from matplotlib.colors import LogNorm
    lv = (1e-10,1e-5)
    Z1 = ty.eps.values
    Zm = np.ma.array(Z1,mask=np.isnan(Z1))
    h = ax.pcolormesh(ty.dist,ty.z,Zm,
                      norm=LogNorm(vmin=lv[0], vmax=lv[-1]),
                      cmap=cmocean.cm.speed,
                      rasterized=True)
    ht = ax.plot(ty.dist,ty.topo,'k')
    ax.set_xlabel('Distance [km]')
    ax.set_ylim([5200,4050])
    ax.set_xlim([-1,32])
    ax.set_ylabel('Depth [m]')
    return h, ht


# %%
def Fig1Vel(ty,ax,vel_component='v', maxv=0.5, step=0.05):
    """Plot Velocity
    
    Args:
        ty (xarray Dataset): towyo dataset
        ax (ax obj): handle to axis
        
    Returns:
        h (plot object): handle to contourf (u/v) plot
        """
    print('plotting velocity {}-component'.format(vel_component))
#     from matplotlib.colors import LogNorm
#     lv = (1e-10,1e-5)
#     Z1 = ty.eps.values
#     Zm = np.ma.array(Z1,mask=np.isnan(Z1))
    field = np.ma.masked_invalid(ty[vel_component])
    h = ax.contourf(ty.dist,ty.z,field,
                    levels=np.arange(-maxv,maxv+step,step),
                    cmap='RdBu_r',
                   extend='both')
    for c in h.collections:
        c.set_rasterized(True)
        c.set_edgecolor('face')
    
    # contour sg4
    if vel_component == 'u':
        ax.contour(ty.dist, ty.z, ty.gsw_sigma4, levels=ty.sg4bins, colors='0.5', linewidths=0.5)
        ax.contour(ty.dist, ty.z, ty.gsw_sigma4, levels=[45.94], colors='k', linewidths=0.75)
    if vel_component == 'v':
        ax.contour(ty.dist, ty.z, ty.gsw_sigma4, levels=[45.94], colors='k', linewidths=0.75)
    if vel_component == 'w':
        ax.contour(ty.dist, ty.z, ty.gsw_sigma4, levels=ty.sg4bins, colors='0.5', linewidths=0.5)
        ax.contour(ty.dist, ty.z, ty.gsw_sigma4, levels=[45.94], colors='k', linewidths=0.75)
    
    ht = ax.plot(ty.dist,ty.topo,'k')
    ax.set_xlabel('Distance [km]')
    ax.set_ylim([5200,4050])
    ax.set_xlim([-1,32])
    ax.set_ylabel('Depth [m]')
    return h


# %%
if is_notebook:
    ty = a['t12']
    fig, ax = gv.plot.newfig()
    Fig1Vel(ty,ax,'v')

# %%
if is_notebook:
    ty = a['t12']
    fig, ax = gv.plot.newfig()
    Fig1Vel(ty,ax,'w', maxv=0.07, step=0.01)


# %%
def Fig1TPXO(ty, ax):
    """Plot TPXO predicted u and v
    
    Args:
        ty (xarray Dataset): towyo dataset
        ax (ax obj): handle to axis
        
    Returns:
        h (plot object): handle to tpxo plot
    """
    print('plotting tpxo...')
    h = []
    h.append(ax.plot(ty.dist, ty.tpxo_velu*100, mfc=col[0], mec=col[0],
                   marker='o', markersize=2, linestyle=''))
    h.append(ax.plot(ty.dist, ty.tpxo_velv*100, mfc=col[1], mec=col[1],
                   marker='o', markersize=2, linestyle=''))
    # plot legend for u/v, but only for 2014 panel
    tmp = pd.to_datetime(ty.time[0].values)
    if tmp.year>2013:
        ax.text(ty.dist[-1]+1, ty.tpxo_velu[-1]*100, 'u', color=col[0])
        ax.text(ty.dist[-1]+1, ty.tpxo_velv[-1]*100, 'v', color=col[1])
    
    ax.set_xlim([-1,32])
    ax.set_ylim([-3,3])
    ax.set_ylabel('TPXO [cm/s]')
    ax.set_yticks([-2, 0, 2])
    ax.set_xlabel('Distance [km]')
    ax.yaxis.set_ticks_position('none')

    return h


# %%
def Fig1ProfileMarker(ty, ax):
    """Plot profile marker along towyo section
       TODO: Also add timestamps.
       
    Args:
        ty (xarray Dataset): towyo dataset
        ax (ax obj): handle to axis
        
    Returns:
        h (plot object): handle to markers
    """
    print('plotting profile markers...')
    h = []
    h.append(ax.plot(ty.dist,np.ones_like(ty.dist)*-1,
                     marker='|', ms=3, mec='k', mfc='k', color='k', linestyle=''))
    ax.set_xlim([-1,32])
    ax.set_ylim([-1,1])
    return h


# %%
def Fig1PrintTimeStamps(ax, ty, ni):
    time = pd.to_datetime(ty.time.values)
    dist = ty.dist.values
    ax.text(dist[0],0.25,'{}'.format(time[0].year),
            fontweight='bold', ha='left', fontsize=9)
    day = time[0].day
    for i, tid in enumerate(zip(time, dist)):
        ti, d = tid
        if day==ti.day:
            ax.text(d,-0.25,'{}/{}'.format(ti.month, ti.day),
                    fontweight='bold', ha='left')
            day=day+1
        if i in ni:
            ax.text(d,-0.75,'{:02d}:{:02d}'.format(ti.hour, ti.minute),
                    ha='center', fontsize=7)


# %% [markdown]
# The new towyo overview figure should have the following panels with two columns for 2012 and 2014 (with rowspan for each row):
# - time stamps and profile markers (2)
# - $\theta$ and w (4)
# - u (4)
# - v and $\sigma_4$ contours (4)
# - $\epsilon$(4)
# - TPXO u and v (1)
#
# 19 rows total

# %%
fig = plt.figure(figsize=(12,12))
# subplot2grid((shape),(loc),colspan=,rowspan=)
nr = 19
nc = 11
# plot axes
ax1 = plt.subplot2grid((nr, nc), (2, 0), rowspan=4, colspan=5)
ax2 = plt.subplot2grid((nr, nc), (2, 5), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax3 = plt.subplot2grid((nr, nc), (6, 0), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax4 = plt.subplot2grid((nr, nc), (6, 5), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax5 = plt.subplot2grid((nr, nc), (10, 0), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax6 = plt.subplot2grid((nr, nc), (10, 5), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax7 = plt.subplot2grid((nr, nc), (14, 0), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax8 = plt.subplot2grid((nr, nc), (14, 5), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax9 = plt.subplot2grid((nr, nc), (18, 0), colspan=5, sharex=ax1)
ax10 = plt.subplot2grid((nr, nc), (18, 5), colspan=5, sharex=ax1, sharey=ax9)
# colorbar axes
cbax1 = plt.subplot2grid((nr, nc), (2, 10), rowspan=4)
cbax2 = plt.subplot2grid((nr, nc), (6, 10), rowspan=4)
cbax3 = plt.subplot2grid((nr, nc), (10, 10), rowspan=4)
cbax4 = plt.subplot2grid((nr, nc), (14, 10), rowspan=4)
# time tick axes
tax1 = plt.subplot2grid((nr, nc), (0, 0), rowspan=2, colspan=5, sharex=ax1)
tax2 = plt.subplot2grid((nr, nc), (0, 5), rowspan=2, colspan=5, sharex=ax1, sharey=tax1)


ax = np.array([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, tax1, tax2])

for axi in ax:
    axi = gv.plot.axstyle(axi)
    
def AdjustColorbar(cb,cbax,lbl):
    cb.ax.set_ylabel(lbl);
    cb.solids.set_rasterized(True) 
    pos1 = cbax.get_position() # get the original position 
    pos2 = [pos1.x0-0.05, pos1.y0+pos1.height/10,  pos1.width / 6.0, pos1.height-pos1.height/5] 
    cbax.set_position(pos2) # set a new position

# theta and w
for kty, axi in zip(a.items(),[ax1, ax2]):
    # ty still is a tuple with (key, value), need to unpack
    k, ty = kty
    h, h2, qk, ht = Fig1THw(ty,'w',axi)
cb = plt.colorbar(h, drawedges=False,
                      ticks=np.arange(0.7,1.2,0.1),cax=cbax1);
AdjustColorbar(cb, cbax1, r'$\theta$ [$^{\circ}$C]')

# v
for kty, axi in zip(a.items(),[ax3, ax4]):
    k, ty = kty
    h = Fig1Vel(ty,axi,'v')
cb = plt.colorbar(h, drawedges=False, cax=cbax2, ticks=np.arange(-0.2,0.6,0.2));
AdjustColorbar(cb, cbax2, r'v [m/s]');

# shear
for kty, axi in zip(a.items(),[ax5, ax6]):
    k, ty = kty
    h = Fig1Shear(ty,axi)
cb = plt.colorbar(h, drawedges=False, cax=cbax3, ticks=np.arange(0,1.2e-5,2e-6), extend='max');
AdjustColorbar(cb, cbax3, r'v$_z^2$ [1/s$^2$]');


# eps
for kty, axi in zip(a.items(),[ax7, ax8]):
    k, ty = kty
    h, ht = Fig1Eps(ty, axi)
cb = plt.colorbar(h, drawedges=False, cax=cbax4);
AdjustColorbar(cb, cbax4, r'$\mathrm{log}_{10}(\epsilon)$ [W/kg]')

# tpxo
for kty, axi in zip(a.items(),[ax9, ax10]):
    k, ty = kty
    Fig1TPXO(ty, axi);

# markers
for kty, axi in zip(a.items(),[tax1, tax2]):
    k, ty = kty
    Fig1ProfileMarker(ty, axi);

# time stamps
ni1 = np.arange(0,len(a['t12']['lon']),10)
ni2 = np.arange(0,len(a['t14']['lon']),15)
for kty, axi, ni in zip(a.items(),[tax1, tax2], [ni1, ni2]):
    k, ty = kty
    Fig1PrintTimeStamps(axi, ty, ni);
    
# no grid lines / spines / ticklabels for some axes
for axi in ax:
    axi.grid(False)
for axi in [tax1, tax2, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    axi.xaxis.set_visible(False)
    axi.spines['bottom'].set_visible(False)
for axi in [tax1, tax2, ax2, ax4, ax6, ax8, ax9, ax10]:
    axi.yaxis.set_visible(False)
    axi.spines['left'].set_visible(False)
ax10.yaxis.set_visible(True)
ax10.yaxis.tick_right()
ax10.yaxis.label_position='right'
ax10.yaxis.labelpad = 20
ax10.spines['right'].set_visible(True)


SubplotLetters = ['a', 'b', 'c', 'd', 'e']
for  letter, axi in zip(SubplotLetters, [ax1, ax3, ax5, ax7, ax9]):
    axi.text(0, 0.97, letter, transform=axi.transAxes,
              horizontalalignment='left', fontweight='bold')

print('saving figure...')
nsl.io.save_png("towyo_overview")
nsl.io.save_pdf("towyo_overview")

# %% [markdown]
# The towyo overview figure for the revised version should have the following panels with two columns for 2012 and 2014 (with rowspan for each row):
# - time stamps and profile markers (2)
# - $\theta$ (4)
# - v (4)
# - w (4)
# - shear and $\sigma_4$ contours (4)
# - $\epsilon$(4)
# - TPXO u and v (1)
#
# 23 rows total

# %%
fig = plt.figure(figsize=(12,14))
# subplot2grid((shape),(loc),colspan=,rowspan=)
nr = 23
nc = 11
# plot axes
ax1 = plt.subplot2grid((nr, nc), (2, 0), rowspan=4, colspan=5)
ax2 = plt.subplot2grid((nr, nc), (2, 5), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax3 = plt.subplot2grid((nr, nc), (6, 0), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax3a = plt.subplot2grid((nr, nc), (6+4, 0), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax4 = plt.subplot2grid((nr, nc), (6, 5), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax4a = plt.subplot2grid((nr, nc), (6+4, 5), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax5 = plt.subplot2grid((nr, nc), (10+4, 0), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax6 = plt.subplot2grid((nr, nc), (10+4, 5), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax7 = plt.subplot2grid((nr, nc), (14+4, 0), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax8 = plt.subplot2grid((nr, nc), (14+4, 5), rowspan=4, colspan=5, sharey=ax1, sharex=ax1)
ax9 = plt.subplot2grid((nr, nc), (18+4, 0), colspan=5, sharex=ax1)
ax10 = plt.subplot2grid((nr, nc), (18+4, 5), colspan=5, sharex=ax1, sharey=ax9)
# colorbar axes
cbax1 = plt.subplot2grid((nr, nc), (2, 10), rowspan=4)
cbax2 = plt.subplot2grid((nr, nc), (6, 10), rowspan=4)
cbax2a = plt.subplot2grid((nr, nc), (6+4, 10), rowspan=4)
cbax3 = plt.subplot2grid((nr, nc), (10+4, 10), rowspan=4)
cbax4 = plt.subplot2grid((nr, nc), (14+4, 10), rowspan=4)
# time tick axes
tax1 = plt.subplot2grid((nr, nc), (0, 0), rowspan=2, colspan=5, sharex=ax1)
tax2 = plt.subplot2grid((nr, nc), (0, 5), rowspan=2, colspan=5, sharex=ax1, sharey=tax1)


ax = np.array([ax1, ax2, ax3, ax4, ax3a, ax4a, ax5, ax6, ax7, ax8, ax9, ax10, tax1, tax2])

for axi in ax:
    axi = gv.plot.axstyle(axi)
    
def AdjustColorbar(cb,cbax,lbl):
    cb.ax.set_ylabel(lbl);
    cb.solids.set_rasterized(True) 
    pos1 = cbax.get_position() # get the original position 
    pos2 = [pos1.x0-0.055, pos1.y0+pos1.height/10,  pos1.width / 10.0, pos1.height-pos1.height/5] 
    cbax.set_position(pos2) # set a new position

# theta
for kty, axi in zip(a.items(),[ax1, ax2]):
    # ty still is a tuple with (key, value), need to unpack
    k, ty = kty
    h, h2, qk, ht = Fig1THw(ty,'w',axi)
cb = plt.colorbar(h, drawedges=False,
                      ticks=np.arange(0.7,1.2,0.1),cax=cbax1);
AdjustColorbar(cb, cbax1, r'$\theta$ [$^{\circ}$C]')

# v
for kty, axi in zip(a.items(),[ax3, ax4]):
    k, ty = kty
    h = Fig1Vel(ty,axi,'v')
cb = plt.colorbar(h, drawedges=False, cax=cbax2, ticks=np.arange(-0.2,0.6,0.2));
AdjustColorbar(cb, cbax2, r'v [m$\,$s$^{-1}$]');

# w
for kty, axi in zip(a.items(),[ax3a, ax4a]):
    k, ty = kty
    h = Fig1Vel(ty,axi,'w', maxv=0.07, step=0.01)
cb = plt.colorbar(h, drawedges=False, cax=cbax2a, ticks=np.arange(-0.06, 0.09, 0.03));
AdjustColorbar(cb, cbax2a, r'w [m$\,$s$^{-1}$]');

# shear
for kty, axi in zip(a.items(),[ax5, ax6]):
    k, ty = kty
    h = Fig1Shear(ty,axi)
cb = plt.colorbar(h, drawedges=False, cax=cbax3, ticks=np.arange(0,1.2e-5,2e-6), extend='max');
AdjustColorbar(cb, cbax3, r'v$_z^2$ [s$^{-2}$]');

# eps
for kty, axi in zip(a.items(),[ax7, ax8]):
    k, ty = kty
    h, ht = Fig1Eps(ty, axi)
cb = plt.colorbar(h, drawedges=False, cax=cbax4);
AdjustColorbar(cb, cbax4, r'$\mathrm{log}_{10}(\epsilon)$ [W$\,$kg$^{-1}$]')

# tpxo
for kty, axi in zip(a.items(),[ax9, ax10]):
    k, ty = kty
    Fig1TPXO(ty, axi);

# markers
for kty, axi in zip(a.items(),[tax1, tax2]):
    k, ty = kty
    Fig1ProfileMarker(ty, axi);

# time stamps
ni1 = np.arange(0,len(a['t12']['lon']),10)
ni2 = np.arange(0,len(a['t14']['lon']),15)
for kty, axi, ni in zip(a.items(),[tax1, tax2], [ni1, ni2]):
    k, ty = kty
    Fig1PrintTimeStamps(axi, ty, ni);
    
# no grid lines / spines / ticklabels for some axes
for axi in ax:
    axi.grid(False)
for axi in [tax1, tax2, ax1, ax2, ax3, ax4, ax3a, ax4a, ax5, ax6, ax7, ax8]:
    axi.xaxis.set_visible(False)
    axi.spines['bottom'].set_visible(False)
for axi in [tax1, tax2, ax2, ax4, ax4a, ax6, ax8, ax9, ax10]:
    axi.yaxis.set_visible(False)
    axi.spines['left'].set_visible(False)
ax10.yaxis.set_visible(True)
ax10.yaxis.tick_right()
ax10.yaxis.label_position='right'
ax10.yaxis.labelpad = 20
ax10.spines['right'].set_visible(True)


SubplotLetters = ['a', 'b', 'c', 'd', 'e', 'f']
for  letter, axi in zip(SubplotLetters, [ax1, ax3, ax3a, ax5, ax7, ax9]):
    axi.text(0, 0.97, letter, transform=axi.transAxes,
              horizontalalignment='left', fontweight='bold')

print('saving figure...')
nsl.io.save_png("towyo_overview")
nsl.io.save_pdf("towyo_overview")

# %%
